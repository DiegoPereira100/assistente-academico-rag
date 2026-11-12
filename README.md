# Assistente Acadêmico — Sistema RAG com LangChain

Projeto acadêmico desenvolvido para a disciplina de **Processamento de Linguagem Natural (PLN)**.

## 1. Objetivo

Desenvolver um assistente acadêmico capaz de responder perguntas sobre regulamentos e documentos institucionais utilizando a arquitetura RAG (*Retrieval-Augmented Generation*).

O sistema consulta os documentos inseridos na base de conhecimento, recupera os trechos relevantes e utiliza um modelo de linguagem para elaborar respostas fundamentadas nas informações encontradas.

Quando não houver evidências suficientes nos documentos, o sistema deve informar que não encontrou informações para responder à pergunta.

## 2. Tecnologias utilizadas

- **Python 3.12:** linguagem principal.
- **LangChain:** integração e organização do fluxo RAG.
- **ChromaDB:** armazenamento vetorial.
- **HuggingFace Sentence Transformers:** geração de embeddings.
- **BM25:** recuperação lexical de documentos.
- **Ollama:** execução local do modelo de linguagem.
- **Qwen2.5:3b:** modelo utilizado para geração das respostas.

## 3. Funcionamento do sistema

O projeto utiliza as seguintes etapas:

1. **Ingestão:** leitura dos documentos PDF.
2. **Pré-processamento:** preparação e organização dos textos extraídos.
3. **Divisão em chunks:** fragmentação dos documentos em trechos menores.
4. **Embeddings:** transformação dos textos em representações vetoriais.
5. **Armazenamento:** persistência dos vetores no ChromaDB.
6. **Recuperação:** busca híbrida, combinando similaridade semântica e BM25.
7. **Geração:** envio dos trechos recuperados ao modelo Qwen2.5:3b por meio do LangChain.
8. **Resposta:** apresentação da resposta e das páginas utilizadas como fontes.

O sistema também possui regras específicas para responder a determinadas perguntas diretamente com base nos artigos recuperados do regulamento.

## 4. Estrutura do projeto

```text
assistente_academico_rag/
├── data/
│   └── pdfs/
│       ├── regulamento_geral_fatecs.pdf
│       └── COLOQUE_OS_PDFS_AQUI.txt
├── avaliacao/
│   ├── avaliar.py
│   ├── perguntas.csv
│   └── resultados.csv
├── chroma_db/                 # Banco vetorial gerado pela ingestão
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   └── chat.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Configuração do ambiente

Requisitos:

- Python 3.12.
- Ollama instalado.
- Modelo Qwen2.5:3b disponível no Ollama.

No PowerShell, dentro da pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instale o modelo utilizado:

```powershell
ollama pull qwen2.5:3b
```

Certifique-se de que o Ollama esteja em execução antes de iniciar o sistema.

## 6. Execução

O documento utilizado como base de conhecimento deve estar disponível na pasta `data/pdfs/`.

Para processar os documentos e criar a base vetorial:

```powershell
python -m src.ingest
```

Para iniciar a interface de perguntas:

```powershell
python -m src.chat
```

Digite as perguntas no terminal e acompanhe as respostas e os trechos recuperados.

## 7. Configuração dos chunks

A configuração principal utilizada no processamento dos documentos é:

- Tamanho do chunk: 1200 caracteres.
- Sobreposição (*overlap*): 200 caracteres.

O projeto também mantém uma configuração experimental definida no arquivo `src/config.py`, com 500 caracteres por chunk e sobreposição de 80 caracteres.

Essa segunda configuração está disponível para experimentos futuros. Não foi realizada uma comparação quantitativa entre as duas configurações nesta avaliação.

## 8. Avaliação

O sistema foi avaliado por meio de um conjunto de 24 perguntas, armazenadas em `avaliacao/perguntas.csv`.

O conjunto contempla perguntas cuja resposta está presente no regulamento e perguntas sobre assuntos que não são abordados pelo documento.

A avaliação foi executada pelo módulo:

```powershell
python -m avaliacao.avaliar
```

Os resultados são registrados automaticamente em `avaliacao/resultados.csv`, contendo a pergunta, a resposta de referência, a resposta gerada, a fonte esperada, as páginas recuperadas e o tipo de resposta.

### Resultados obtidos

Nos testes realizados:

- 20 perguntas possuíam informações disponíveis no regulamento e receberam respostas fundamentadas no conteúdo recuperado.
- 4 perguntas tratavam de informações ausentes na base e receberam respostas de abstenção, informando que o regulamento não continha os dados solicitados.
- Não foram observadas falhas de execução durante essa avaliação.

Assim, as 24 perguntas apresentaram o comportamento esperado no conjunto de testes utilizado.

A avaliação foi analisada manualmente, considerando a fidelidade ao regulamento, a identificação das fontes e a capacidade de reconhecer a ausência de informações.

O campo `avaliacao` do arquivo CSV indica a necessidade de revisão manual e não representa uma pontuação automática de acerto.

## 9. Documentos utilizados

A base de conhecimento utiliza o documento:

**Regulamento Geral dos Cursos Superiores de Graduação das Faculdades de Tecnologia (Fatecs)**, aprovado pela Deliberação CEETEPS nº 106, de 13 de novembro de 2025, consolidada com a Deliberação CEETEPS nº 117, de 18 de junho de 2026.

- **Instituição responsável:** Centro Estadual de Educação Tecnológica Paula Souza (CEETEPS).
- **Publicação:** Diário Oficial do Estado de São Paulo, volume 135, nº 228, Caderno Executivo, Seção 1, em 18 de novembro de 2025, página 29.
- **Extensão:** 35 páginas.
- **Arquivo:** `regulamento_geral_fatecs.pdf`.

O documento foi escolhido por reunir normas institucionais relacionadas à vida acadêmica dos estudantes das Fatecs, permitindo consultas sobre frequência, avaliação, matrícula e outros procedimentos acadêmicos.

## 10. Observações

- O modelo de linguagem é executado localmente por meio do Ollama.
- Não é necessária uma chave de API para utilizar o modelo configurado.
- O banco vetorial é persistido localmente na pasta `chroma_db`.
- A base vetorial é criada durante a execução do processo de ingestão.
- Os documentos utilizados como fonte devem estar disponíveis na pasta `data/pdfs/`.
- O projeto tem finalidade acadêmica e utiliza recuperação de informações para fundamentar as respostas.