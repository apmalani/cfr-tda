from __future__ import annotations

import pyspiel
import torch

from .game import TensorGame

CARDS = "JQK"
PASS, BET = 0, 1

INFOSET_LABELS = ("J", "Q", "K", "Jpb", "Qpb", "Kpb", "Jb", "Qb", "Kb", "Jp", "Qp", "Kp")
DEAL_LABELS = ("JQ", "JK", "QJ", "QK", "KJ", "KQ")


def openspiel_key_to_label(key: str) -> str:
    return CARDS[int(key[0])] + key[1:]


def label_to_openspiel_key(label: str) -> str:
    return str(CARDS.index(label[0])) + label[1:]


class Kuhn(TensorGame):

    def __init__(self):
        super().__init__(pyspiel.load_game("kuhn_poker"))
        self.infoset_labels = tuple(openspiel_key_to_label(k) for k in self.infoset_keys)
        self.deal_labels = tuple(CARDS[a] + CARDS[b] for a, b in self.deals)
        assert sorted(self.infoset_labels) == sorted(INFOSET_LABELS)
        assert sorted(self.deal_labels) == sorted(DEAL_LABELS)

    def infoset(self, label: str) -> int:
        return self.infoset_index[label_to_openspiel_key(label)]

    def deal(self, label: str) -> int:
        return self.deal_labels.index(label)

