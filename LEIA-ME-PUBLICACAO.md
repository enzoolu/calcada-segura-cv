# Publicar no GitHub e enviar no Teams

O ZIP contém os arquivos descompactados do projeto. Não contém a pasta `.git`, credenciais ou ambiente virtual. A publicação será feita no seu computador; o assistente não confirmou um novo commit remoto.

1. Clone `https://github.com/enzoolu/calcada-segura-cv.git`, se ainda não tiver uma cópia local. Se já tiver, atualize-a antes de extrair os arquivos e preserve alterações locais que não façam parte desta entrega.
2. Extraia **o conteúdo do ZIP dentro da pasta clonada**. Nessa pasta devem ficar `README.md`, `visao/` e `integracao/`, ao lado da pasta `.git` que o clone criou. Evite criar uma segunda pasta de projeto dentro da primeira.
3. Abra essa pasta no VS Code e execute:

```powershell
git status
git add .
git commit -m "Conclui entrega de Computer Vision com treino, avaliacao e relatorio"
git push origin main
```

Se ainda precisar clonar, antes da extração:

```powershell
git clone https://github.com/enzoolu/calcada-segura-cv.git
cd calcada-segura-cv
```

4. Depois de concluir o push, abra os links em janela anônima:

- Repositório: https://github.com/enzoolu/calcada-segura-cv
- PDF: https://github.com/enzoolu/calcada-segura-cv/blob/main/visao/docs/relatorio-visao.pdf
- Vídeo: https://github.com/enzoolu/calcada-segura-cv/blob/main/visao/evidencias/demonstracao-visao.mp4
- Pesos principais: https://github.com/enzoolu/calcada-segura-cv/blob/main/visao/evidencias/treino-publico/best.pt
- Novo especialista experimental: https://github.com/enzoolu/calcada-segura-cv/blob/main/visao/evidencias/roboflow-v6/best.pt

5. Abra ou baixe o vídeo e confira se o PDF está acessível. Somente depois envie `relatorio-visao.pdf` na tarefa de Applied Computer Vision no Teams. Os links do PDF apontam para os caminhos acima; ficam completos quando o conteúdo for publicado.

O vídeo principal tem 87,95 segundos e mostra o detector original. O novo especialista não substitui o padrão. O PDF apresenta os resultados reais e as limitações, inclusive os falsos alarmes. Não existe promessa de nota ou de detecção de todos os defeitos.

Todos os arquivos individuais do pacote foram conferidos para ficar abaixo de 100 MB. O ZIP é um meio de transporte: publique os arquivos extraídos, em vez de enviar somente o ZIP ao GitHub.
