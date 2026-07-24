from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AnswerGenerator:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(self.device)
        # Clear sampling defaults inherited from the model configuration.
        # Sampling values are supplied explicitly only when do_sample=True.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        self.model.eval()

    @staticmethod
    def _build_messages(question: str, contexts: list[str]) -> list[dict[str, str]]:
        evidence = "\n\n".join(
            f"[Document {index + 1}]\n{context}"
            for index, context in enumerate(contexts)
        )

        return [
            {
                "role": "system",
                "content": (
                    "You answer evidence-grounded question-answering tasks. "
                    "Use only the supplied evidence. Return only the shortest "
                    "answer span that directly answers the question. "
                    "Do not explain your reasoning, repeat the question, cite "
                    "document numbers, or add complete sentences unless needed. "
                    'If the evidence is insufficient, return exactly: "I do not know".'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{evidence}\n\n"
                    f"Question: {question}\n\n"
                    "Return only the short answer:"
                ),
            },
        ]

    @staticmethod
    def clean_answer(raw_answer: str) -> str:
        answer = raw_answer.strip()
        answer = re.sub(
            r"^(?:final\s+answer|short\s+answer|answer)\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        non_empty_lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if non_empty_lines:
            answer = non_empty_lines[0]

        answer = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", answer).strip()
        answer = answer.strip(" \"'`")
        return answer

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        contexts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, str]:
        messages = self._build_messages(question, contexts)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95
            generation_kwargs["top_k"] = 50

        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        raw_answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        cleaned_answer = self.clean_answer(raw_answer)
        return raw_answer, cleaned_answer
