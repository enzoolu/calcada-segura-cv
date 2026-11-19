import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx

from src.inferir import reenviar


def evento(pasta, nome):
    caminho = pasta / f"{nome}.json"
    caminho.write_text(json.dumps({"id": nome, "tipo": "calcada_quebrada"}), encoding="utf-8")
    return caminho


class TesteFila(unittest.TestCase):
    def test_rejeicao_de_dados_isola_o_evento_e_envia_o_proximo(self):
        chamadas = []

        def responder(request):
            chamadas.append(str(request.url.path))
            return httpx.Response(422 if len(chamadas) == 1 else 201)

        with TemporaryDirectory() as pasta, httpx.Client(
            base_url="http://local.test/", transport=httpx.MockTransport(responder),
        ) as cliente:
            fila = Path(pasta)
            recusado, valido = evento(fila, "a"), evento(fila, "b")
            original = recusado.read_bytes()
            reenviar(cliente, fila)
            self.assertFalse(valido.exists())
            self.assertFalse(recusado.exists())
            self.assertEqual(original, (fila / "revisao" / recusado.name).read_bytes())
            self.assertEqual(["/ocorrencias", "/ocorrencias"], chamadas)

    def test_falhas_de_servico_ou_credencial_preservam_a_fila(self):
        for status in (None, 401, 503):
            with self.subTest(status=status), TemporaryDirectory() as pasta:
                chamadas = []

                def responder(request):
                    chamadas.append(request)
                    if status is None:
                        raise httpx.ConnectError("API desligada", request=request)
                    return httpx.Response(status)

                fila = Path(pasta)
                a, b = evento(fila, "a"), evento(fila, "b")
                esperados = [p.read_bytes() for p in (a, b)]
                with httpx.Client(base_url="http://local.test/",
                                  transport=httpx.MockTransport(responder)) as cliente:
                    reenviar(cliente, fila)
                self.assertEqual(1, len(chamadas))
                self.assertEqual(esperados, [p.read_bytes() for p in (a, b)])
                self.assertFalse((fila / "revisao").exists())

    def test_evento_invalido_localmente_nao_impede_o_envio_valido(self):
        chamadas = []

        def responder(request):
            chamadas.append(request)
            return httpx.Response(201)

        with TemporaryDirectory() as pasta, httpx.Client(
            base_url="http://local.test/", transport=httpx.MockTransport(responder),
        ) as cliente:
            fila = Path(pasta)
            (fila / "a.json").write_text("{")
            (fila / "b.json").write_text('{"foto_jpeg":"!"}')
            valido = evento(fila, "c")
            reenviar(cliente, fila)
            self.assertFalse(valido.exists())
            self.assertEqual(["a.json", "b.json"], sorted(p.name for p in (fila / "revisao").iterdir()))
            self.assertEqual(1, len(chamadas))


if __name__ == "__main__":
    unittest.main()
