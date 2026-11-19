from collections import deque
import unittest

import numpy as np
import torch
from ultralytics.engine.results import Results

from src.inferir import detecoes_confirmadas, filtrar_deteccoes


class TesteFiltro(unittest.TestCase):
    def test_faixa_vertical_remove_apenas_obstaculos_muito_baixos(self):
        quadro = np.zeros((100, 100, 3), dtype=np.uint8)
        caixa_proxima = [10, 75, 60, 100, .85]
        resultado = Results(quadro, "quadro", {0: "calcada_quebrada", 1: "obstaculo"}, boxes=torch.tensor([
            [*caixa_proxima, 0], [*caixa_proxima, 1], [10, 20, 30, 50, .77, 1],
        ]))
        limiares = {"calcada_quebrada": .3, "obstaculo": .5}
        self.assertEqual(len(filtrar_deteccoes(resultado, limiares)), 3)
        filtrado = filtrar_deteccoes(resultado, limiares, obstaculo_ate_y=.75)
        self.assertEqual(filtrado.boxes.cls.tolist(), [0, 1])
        self.assertEqual(filtrado.boxes.xyxy[1].tolist(), [10, 20, 30, 50])

    def test_separa_buracos_da_saida_das_demais_classes(self):
        quadro = np.zeros((32, 32, 3), dtype=np.uint8)
        caixas = torch.tensor([
            [2, 2, 6, 6, 0.20, 0],
            [5, 5, 7, 7, 0.10, 0],
            [8, 8, 12, 12, 0.60, 1],
            [15, 15, 21, 21, 0.80, 2],
        ])
        resultado = Results(quadro, "quadro", {0: "buraco", 1: "calcada_quebrada", 2: "obstaculo"},
                            boxes=caixas)
        principal = filtrar_deteccoes(resultado, {"calcada_quebrada": 0.30, "obstaculo": 0.50})
        buracos = filtrar_deteccoes(resultado, {"buraco": 0.12})
        self.assertEqual(principal.boxes.cls.tolist(), [1, 2])
        self.assertEqual(buracos.boxes.cls.tolist(), [0])


class TesteConfirmacao(unittest.TestCase):
    def test_confirma_caixas_proximas_em_tres_quadros(self):
        quadros = deque([
            {"buraco": [(0.8, [0.20, 0.20, 0.50, 0.60])]},
            {"buraco": [(0.8, [0.22, 0.22, 0.53, 0.62])]},
            {"buraco": [(0.7, [0.24, 0.24, 0.55, 0.62])]},
        ])
        self.assertEqual(list(detecoes_confirmadas(quadros)), [("buraco", 0.7)])

    def test_mesma_classe_em_regioes_distintas_nao_gera_evento(self):
        quadros = deque([
            {"buraco": [(0.9, [0.10, 0.10, 0.20, 0.30])]},
            {"buraco": [(0.7, [0.20, 0.20, 0.50, 0.60])]},
            {"buraco": [(0.6, [0.21, 0.23, 0.50, 0.62])]},
        ])
        self.assertEqual(list(detecoes_confirmadas(quadros)), [])

    def test_intervalo_sem_deteccao_interrompe_confirmacao(self):
        quadros = deque([
            {"obstaculo": [(0.9, [0.1, 0.1, 0.2, 0.3])]},
            {},
            {"obstaculo": [(0.8, [0.1, 0.1, 0.2, 0.3])]},
        ])
        self.assertEqual(list(detecoes_confirmadas(quadros)), [])
