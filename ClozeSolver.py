from dataclasses import dataclass, field
from typing import Dict, List, Set, Iterable, Optional
import json
import numpy as np
from collections import defaultdict

from ClozeUtils import (
    SENTENCE_START_SYMBOL,
    SENTENCE_END_SYMBOL,
    get_all_contexts,
    clean_line,
)

@dataclass
class ClozeSolver:
    # Core configuration
    k: float = 0.02
    left_only: bool = False
    n: int = 2

    # Paths
    #optional - you don't need to set these for initialization of an instance
    corpus_path: Optional[str] = None
    input_path: Optional[str] = None
    candidates_path: Optional[str] = None

    # Candidates
    #field(default_factory=list) - each instance of the class will have its own independent list,
    #If you don’t provide a value, the default will be used automatically
    candidates: List[str] = field(default_factory=list)
    candidates_set: Set[str] = field(default_factory=set)

    # Model state
    #the lambda ensures that every single instance gets a unique defaultdict
    word2freq: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vocab_size: int = 0 # unique words (types)

    # -------------------------------------------------
    # Construction helpers
    # -------------------------------------------------
    @classmethod
    def from_config(self, config_path: str = "config.json") -> "ClozeSolver":
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        # the actual creation of an instance
        return self(
            k=cfg.get('k', 0.02),
            left_only=cfg.get('left_only', False),
            n=cfg.get('n', 2),
            corpus_path=cfg.get('corpus_filename'),
            input_path=cfg.get('input_filename'),
            candidates_path=cfg.get('candidates_filename')
        )

    def load_candidates(self, candidates_path:Optional[str] = None) -> None:

        # if the function call specified a new path update the attribute
        if candidates_path: #checks parameter is not None
            self.candidates_path = candidates_path

        if not self.candidates_path: #checks if the attribute is set
            raise ValueError("candidate_path is not set")

        with open(candidates_path, 'r', encoding='utf-8') as f:
            self.candidates = [line.strip() for line in f if line.strip()]
        self.candidates_set = set(self.candidates)

    # -------------------------------------------------
    # Training: extract n-gram counts around candidate windows
    # -------------------------------------------------
    def train(self, corpus_path: Optional[str] = None) -> None:

        #if the function call specified a new path update the attribute
        if corpus_path: #checks parameter is not None
            self.corpus_path = corpus_path


        if not self.corpus_path:
            raise ValueError("corpus_path is not set")
        if not self.candidates_set:

            raise ValueError("candidates are not set; call load_candidates()")

        print(f"[train] Start training from '{self.corpus_path}' with {len(self.candidates)} candidates...")

        word2freq = defaultdict(int)
        with open(self.corpus_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                cleaned_line = clean_line(line).split()
                padded_line = [SENTENCE_START_SYMBOL] + cleaned_line + [SENTENCE_END_SYMBOL]
                line_len = len(padded_line)
                for i in range(1, line_len - 1):
                    w = padded_line[i]
                    if w not in self.candidates_set:
                        continue
                    w_prev = padded_line[i - 1]
                    w_next = padded_line[i + 1]

                    # unigrams
                    word2freq[w_prev] += 1
                    word2freq[w] += 1
                    word2freq[w_next] += 1

                    # bigrams
                    word2freq[f"{w_prev} {w}"] += 1
                    word2freq[f"{w} {w_next}"] += 1

                    # left trigram
                    if i - 2 >= 0:
                        w_prev2 = padded_line[i - 2]
                        word2freq[w_prev2] += 1
                        word2freq[f"{w_prev2} {w_prev} {w}"] += 1

                    # right trigram
                    if i + 2 < line_len:
                        w_next2 = padded_line[i + 2]
                        word2freq[w_next2] += 1
                        word2freq[f"{w} {w_next} {w_next2}"] += 1

                    # middle trigram
                    if w_prev != SENTENCE_START_SYMBOL and w_next != SENTENCE_END_SYMBOL:
                        word2freq[f"{w_prev} {w} {w_next}"] += 1

        print(f"[train] Done. {len(word2freq):,} total ngram keys.")
        self.word2freq = word2freq
        self.vocab_size = len({k for k in word2freq.keys() if ' ' not in k})

    # -------------------------------------------------
    # Prediction for a single blank
    # -------------------------------------------------
    def predict_one(self, context: List[str]) -> str:
        if self.vocab_size == 0:
            return self.candidates[0] if self.candidates else ""

        w_b2, w_b1, w_a1, w_a2 = None, None, None, None

        if self.left_only:
            w_b2, w_b1= context
        else:
            w_b2, w_b1, w_a1, w_a2 = context

        best_cand = self.candidates[0] if self.candidates else ""
        best = -1.0

        for cand in self.candidates:
            # left trigram P(cand | w_b2, w_b1)
            left_hist = f"{w_b2} {w_b1}"
            left_3 = f"{w_b2} {w_b1} {cand}"
            p_left = (self.word2freq.get(left_3, 0) + self.k) / (self.word2freq.get(left_hist, 0) + self.k * self.vocab_size)

            # right trigram P(w_a2 | cand, w_a1)
            right_hist = f"{cand} {w_a1}"
            right_3 = f"{cand} {w_a1} {w_a2}"
            p_right = (self.word2freq.get(right_3, 0) + self.k) / (self.word2freq.get(right_hist, 0) + self.k * self.vocab_size)

            # middle trigram P(w_a1 | w_b1, cand)
            mid_hist = f"{w_b1} {cand}"
            mid_3 = f"{w_b1} {cand} {w_a1}"
            p_mid = (self.word2freq.get(mid_3, 0) + self.k) / (self.word2freq.get(mid_hist, 0) + self.k * self.vocab_size)

            # same decision rule as your function (take max)
            score = max(p_left, p_right, p_mid)
            if score > best:
                best = score
                best_cand = cand
        return best_cand

    # -------------------------------------------------
    # Batch prediction
    # -------------------------------------------------
    def predict_many(self, contexts: List[List[str]]) -> List[str]:
        return [self.predict_one(ctx) for ctx in contexts]

    # -------------------------------------------------
    # End-to-end solving
    # -------------------------------------------------
    def solve_helper(self, input_path: Optional[str] = None) -> List[str]:
        if input_path:
            self.input_path = input_path
        if not self.input_path:
            raise ValueError("input_path is not set")

        print(f"[solveHelper] Input='{self.input_path}', left_only={self.left_only}, n={self.n}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        contexts = get_all_contexts(text, n=self.n, left=self.left_only)
        print(f"[solveHelper] Contexts: {len(contexts)}; starting prediction...")

        print("[solveHelper] Prediction done.")
        return self.predict_many(contexts)

    @classmethod
    def solve(cls, config_path: str = "config.json") -> List[str]:
        """Convenience entrypoint: build a solver from config and run end-to-end.
        Allows calling `ClozeSolver.solve()` directly from `main.py`.
        """
        solver = cls.from_config(config_path)

        solver.load_candidates(solver.candidates_path)

        solver.train()  # uses paths and candidates already set

        return solver.solve_helper()  # uses input_path from config
