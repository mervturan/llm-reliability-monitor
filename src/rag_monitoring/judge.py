from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class JudgeResult:
    faithfulness: int
    answer_relevance: int
    reasoning: str
    raw_response: str
    parse_success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgeEvaluator:
    """
    Evaluate a generated RAG answer using an instruction-tuned LLM.

    The judge scores:
      - faithfulness to the retrieved evidence;
      - relevance to the question.

    Each score ranges from 1 to 5.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": "auto",
        }

        if device == "auto":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

    def build_prompt(
        self,
        question: str,
        contexts: list[str],
        prediction: str,
    ) -> str:
        numbered_contexts = "\n\n".join(
            f"[Context {index}]\n{context}"
            for index, context in enumerate(
                contexts,
                start=1,
            )
        )

        return f"""
        You are evaluating an answer produced by a
        Retrieval-Augmented Generation (RAG) system.

        QUESTION:
        {question}

        RETRIEVED CONTEXTS:
        {numbered_contexts}

        CANDIDATE ANSWER:
        {prediction}

        Evaluate the candidate answer on two separate criteria.

        FAITHFULNESS:
        Assess whether the factual claims in the candidate answer are
        supported by the retrieved contexts.

        1 = unsupported or contradicted by the retrieved evidence
        2 = mostly unsupported
        3 = partially supported
        4 = mostly supported
        5 = fully supported by the retrieved evidence

        Support from one relevant retrieved context is sufficient.
        Do not penalise the answer because other retrieved contexts are
        irrelevant.

        If a factual claim in the candidate answer is explicitly stated in,
        or can be directly inferred from, at least one retrieved context,
        treat that claim as supported.

        ANSWER RELEVANCE:
        Assess only whether the candidate answer attempts to answer the
        question that was asked.

        Do NOT use the retrieved contexts when scoring answer relevance.
        An answer may be relevant to the question even if it is factually
        incorrect or unsupported by the retrieved evidence.

        Examples:
        Question: "When did X happen?"
        Answer: "1962"
        This is relevant to the question because it provides a date, even
        if the date is incorrect or unsupported.

        Question: "Who founded X?"
        Answer: "John Smith"
        This is relevant because it provides a person, even if that person
        is incorrect.

        Incorrectness or lack of evidence should affect faithfulness, not
        answer relevance.

        Responses such as "I do not know", refusals, or answers that provide
        no information requested by the question should receive low answer
        relevance.

        1 = does not answer the question
        2 = mostly irrelevant
        3 = partially answers the question
        4 = mostly relevant
        5 = directly and appropriately answers the question

        Important rules:
        - Evaluate only the candidate answer.
        - Do not assume the candidate is good merely because relevant
        information appears in the retrieved contexts.
        - For faithfulness, judge only whether claims made by the candidate
        are supported by the retrieved evidence.
        - For answer relevance, judge whether the candidate itself addresses
        the question.
        - Score faithfulness and answer relevance independently.
        - Return only one valid JSON object.
        - Do not output Markdown or text outside the JSON object.

        Return exactly this structure:
        {{
        "faithfulness": 1,
        "answer_relevance": 1,
        "reasoning": "One concise sentence explaining the scores."
        }}
        """.strip()

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 160,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        model_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = generated[
            :,
            inputs["input_ids"].shape[1] :
        ]

        return self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _extract_json(
        response: str,
    ) -> dict[str, Any]:
        """
        Parse a JSON response, with a fallback that extracts the first
        JSON object if the model adds extra text.
        """

        try:
            parsed = json.loads(response)

        except json.JSONDecodeError:
            match = re.search(
                r"\{.*\}",
                response,
                flags=re.DOTALL,
            )

            if match is None:
                raise ValueError(
                    "Judge response did not contain a JSON object."
                )

            parsed = json.loads(
                match.group(0)
            )

        if not isinstance(parsed, dict):
            raise ValueError(
                "Judge response must be a JSON object."
            )

        return parsed

    @staticmethod
    def _validate_score(
        value: Any,
        field_name: str,
    ) -> int:
        try:
            score = int(value)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field_name} must be an integer."
            ) from error

        if score < 1 or score > 5:
            raise ValueError(
                f"{field_name} must be between 1 and 5."
            )

        return score

    def parse_response(
        self,
        response: str,
    ) -> JudgeResult:
        data = self._extract_json(
            response
        )

        return JudgeResult(
            faithfulness=self._validate_score(
                data.get("faithfulness"),
                "faithfulness",
            ),
            answer_relevance=self._validate_score(
                data.get("answer_relevance"),
                "answer_relevance",
            ),
            reasoning=str(
                data.get(
                    "reasoning",
                    "",
                )
            ).strip(),
            raw_response=response,
            parse_success=True,
        )

    def evaluate(
        self,
        question: str,
        contexts: list[str],
        prediction: str,
        max_new_tokens: int = 160,
    ) -> JudgeResult:
        prompt = self.build_prompt(
            question=question,
            contexts=contexts,
            prediction=prediction,
        )

        response = self.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        try:
            return self.parse_response(
                response
            )

        except (
            ValueError,
            json.JSONDecodeError,
        ):
            repair_prompt = f"""
        The following evaluation contains invalid JSON:

        {response}

        Return the same evaluation again as valid JSON only.

        Use exactly this structure:
        {{
        "faithfulness": 1,
        "answer_relevance": 1,
        "reasoning": "One short sentence without double quotation marks."
        }}

        Do not include Markdown or any text outside the JSON object.
        """.strip()

            repaired_response = self.generate_response(
                prompt=repair_prompt,
                max_new_tokens=max_new_tokens,
            )

            try:
                return self.parse_response(
                    repaired_response
                )

            except (
                ValueError,
                json.JSONDecodeError,
            ):
                return JudgeResult(
                    faithfulness=0,
                    answer_relevance=0,
                    reasoning=(
                        "The judge response could not be parsed "
                        "after one repair attempt."
                    ),
                    raw_response=repaired_response,
                    parse_success=False,
                )