
import csv
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
)

from src.chat import (
    tokenize,
    hybrid_search,
    prepare_context,
    answer_specific_rule,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_INPUT = PROJECT_ROOT / "avaliacao" / "perguntas.csv"
CSV_OUTPUT = PROJECT_ROOT / "avaliacao" / "resultados.csv"


def load_questions():
    with open(
        CSV_INPUT,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.reader(file, delimiter=",")

        rows = [
            row for row in reader
            if row and any(value.strip() for value in row)
        ]

        if not rows:
            raise ValueError("O CSV está vazio.")

        headers = [header.strip() for header in rows[0]]

        required_columns = [
            "pergunta",
            "resposta_referencia",
            "a_base_responde",
            "fonte_ou_observacao",
        ]

        for column in required_columns:
            if column not in headers:
                raise ValueError(
                    f"Coluna obrigatória não encontrada: {column}"
                )

        questions = []

        for row in rows[1:]:
            data = dict(zip(headers, row))

            question = data.get("pergunta", "").strip()

            if not question:
                continue

            questions.append({
                "pergunta": question,
                "referencia": data.get(
                    "resposta_referencia", ""
                ),
                "fonte": data.get(
                    "fonte_ou_observacao", ""
                ),
                "a_base_responde": data.get(
                    "a_base_responde", ""
                ),
            })

    return questions


def create_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é um Assistente Acadêmico especializado
em interpretar regulamentos acadêmicos.

Responda sempre em português, utilizando exclusivamente
as informações presentes no contexto fornecido.

REGRAS OBRIGATÓRIAS:

1. Leia todos os trechos antes de responder.
2. Responda a todas as partes da pergunta.
3. Combine informações de diferentes trechos quando necessário.
4. Não invente informações, regras, prazos ou artigos.
5. Identifique corretamente artigo, parágrafo e inciso.
6. Utilize somente redações atualizadas dos artigos.
7. Diferencie reprovação, aprovação e aproveitamento.
8. Não transforme uma proibição de obter dispensa
   em uma proibição de solicitar aproveitamento.
9. Se o contexto não contiver informação suficiente,
   informe que não foi possível encontrar a informação.
10. Não tente adivinhar respostas.

Contexto:

{context}"""
        ),
        ("human", "{question}"),
    ])


def get_pages(results):
    pages = set()

    for document in results:
        page = document.metadata.get("page")

        if isinstance(page, int):
            pages.add(str(page + 1))
        elif page is not None:
            pages.add(str(page))

    return ", ".join(sorted(pages, key=str))


def main():
    print("\n=== AVALIAÇÃO DO ASSISTENTE ACADÊMICO ===")

    print("\nCarregando perguntas...")
    questions = load_questions()

    print(f"Perguntas encontradas: {len(questions)}")

    print("\nCarregando banco vetorial...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    stored_data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    documents = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            stored_data["documents"],
            stored_data["metadatas"]
        )
        if content
    ]

    if not documents:
        raise ValueError("Nenhum documento encontrado no banco.")

    bm25 = BM25Okapi([
        tokenize(document.page_content)
        for document in documents
    ])

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    prompt = create_prompt()

    resultados = []

    print("\nIniciando avaliação...\n")

    for index, item in enumerate(questions, start=1):
        question = item["pergunta"]

        print("=" * 60)
        print(f"Pergunta {index}/{len(questions)}")
        print(question)

        try:
            results = hybrid_search(
                question,
                vectorstore,
                bm25,
                documents,
                k=6
            )

            context = prepare_context(results, documents)

            pages = get_pages(results)

            # Perguntas cuja resposta não existe na base.
            if item["a_base_responde"].strip().lower() == "nao":
                answer = (
                    "Não foi encontrada informação sobre esse assunto "
                    "no regulamento acadêmico utilizado como base."
                )

                answer_type = "Ausência de informação na base"

            else:
                # Verifica se existe uma resposta determinística.
                direct_answer = answer_specific_rule(
                    question,
                    results
                )

                if direct_answer is not None:
                    answer = direct_answer
                    answer_type = "Regra determinística"

                elif results:
                    response = (prompt | llm).invoke({
                        "context": context,
                        "question": question,
                    })

                    answer = response.content
                    answer_type = "LLM"

                else:
                    answer = (
                        "Não foi encontrada informação relevante "
                        "no regulamento para responder à pergunta."
                    )

                    answer_type = "Sem recuperação"

            print("\nResposta:")
            print(answer)

            print(f"\nPáginas recuperadas: {pages or 'Nenhuma'}")
            print(f"Tipo de resposta: {answer_type}")

            resultados.append({
                "pergunta": question,
                "resposta_referencia": item["referencia"],
                "resposta_gerada": answer,
                "fonte_esperada": item["fonte"],
                "paginas_recuperadas": pages,
                "a_base_responde": item["a_base_responde"],
                "tipo_resposta": answer_type,
                "avaliacao": "REVISAR MANUALMENTE",
            })

        except Exception as error:
            print(f"Erro: {error}")

            resultados.append({
                "pergunta": question,
                "resposta_referencia": item["referencia"],
                "resposta_gerada": f"ERRO: {error}",
                "fonte_esperada": item["fonte"],
                "paginas_recuperadas": "",
                "a_base_responde": item["a_base_responde"],
                "tipo_resposta": "Erro",
                "avaliacao": "ERRO",
            })

    output_columns = [
        "pergunta",
        "resposta_referencia",
        "resposta_gerada",
        "fonte_esperada",
        "paginas_recuperadas",
        "a_base_responde",
        "tipo_resposta",
        "avaliacao",
    ]

    with open(
        CSV_OUTPUT,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_columns
        )

        writer.writeheader()
        writer.writerows(resultados)

    print("\n" + "=" * 60)
    print("AVALIAÇÃO CONCLUÍDA!")
    print(f"Total de perguntas: {len(questions)}")
    print(f"Resultados salvos em: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
