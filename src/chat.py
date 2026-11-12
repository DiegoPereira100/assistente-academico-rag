
import re
import unicodedata

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


def tokenize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    return re.findall(r"\b\w+\b", text)


def normalize(text):
    return " ".join(tokenize(text))


def expand_query(question):
    words = set(tokenize(question))
    expansions = []

    if words.intersection({
        "falta", "faltas", "frequencia",
        "presenca", "ausencia", "ausencias",
        "ausentado"
    }):
        expansions.append(
            "Artigo 39 frequência mínima reprovação "
            "mais de 25% ausências aulas"
        )

    if words.intersection({
        "abono", "militar", "judiciaria",
        "policial", "representacao", "colegiados",
        "paternidade"
    }):
        expansions.append(
            "Artigo 40 abono de faltas serviços militares "
            "autoridade judiciária policial representação "
            "estudantil órgãos colegiados licença paternidade"
        )

    if words.intersection({
        "revisao", "revisar", "media"
    }):
        expansions.append(
            "Artigo 38 revisão da média final prazo aluno "
            "professor igual período três dias consecutivos"
        )

    if words.intersection({
        "exercicio", "domiciliar", "domiciliares"
    }):
        expansions.append(
            "Artigo 42 Artigo 43 regime exercício domiciliar "
            "compensação ausências requerimento cinco dias"
        )

    if words.intersection({
        "rematricula", "rematriculas"
    }):
        expansions.append(
            "Artigo 26 Artigo 32 inciso IX rematrícula "
            "obrigatória responsabilidade aluno manutenção vaga"
        )

    if words.intersection({
        "cancelamento", "cancelada", "cancelado",
        "cancelar"
    }):
        expansions.append(
            "Artigo 32 cancelamento matrícula incisos "
            "ingressante veterano progressão"
        )

    if words.intersection({
        "avaliacao", "corrigida", "professor"
    }):
        expansions.append(
            "Artigo 35 parágrafo 7 dever professor permitir "
            "acesso instrumento avaliação após correção"
        )

    if words.intersection({
        "aprovacao", "aprovar", "aprovado",
        "reprovacao", "reprovar", "reprovado",
        "progressao", "veterano"
    }):
        expansions.append(
            "Artigo 32 incisos II III VIII aprovação "
            "ingressante veterano percentual progressão"
        )

    if expansions:
        return question + " " + " ".join(expansions)

    return question


def identify_articles(text):
    articles = re.findall(
        r"\bArtigo\s+(\d+)\b",
        text,
        flags=re.IGNORECASE
    )

    normalized = normalize(text)

    if (
        "dever do professor permitir ao aluno" in normalized
        and "acesso ao instrumento de" in normalized
    ):
        articles.append("35")

    return sorted(set(articles), key=int)


def is_updated_version(text):
    return bool(
        re.search(
            r"Redação\s+dada\s+pelo",
            text,
            flags=re.IGNORECASE
        )
    )


def remove_obsolete_article_versions(results):
    article_pattern = re.compile(
        r"\bArtigo\s+(\d+)\b",
        flags=re.IGNORECASE
    )

    updated_articles = set()
    split_documents = []

    for document in results:
        content = document.page_content
        matches = list(article_pattern.finditer(content))

        if not matches:
            split_documents.append((document, content, []))
            continue

        prefix = content[:matches[0].start()]
        sections = []

        for index, match in enumerate(matches):
            start = match.start()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            )

            number = match.group(1)
            section = content[start:end]

            sections.append((number, section))

            if is_updated_version(section):
                updated_articles.add(number)

        split_documents.append((document, prefix, sections))

    filtered = []

    for document, prefix, sections in split_documents:
        if not sections:
            filtered.append(document)
            continue

        parts = [prefix]

        for number, section in sections:
            if (
                number in updated_articles
                and not is_updated_version(section)
            ):
                continue

            parts.append(section)

        content = "".join(parts).strip()

        if content:
            filtered.append(
                Document(
                    page_content=content,
                    metadata=document.metadata
                )
            )

    return filtered


def answer_specific_rule(question, results, context=None):
    question_words = set(tokenize(question))

    source_text = context if context else "\n".join(
        document.page_content for document in results
    )

    normalized = normalize(source_text)
    context_words = set(tokenize(source_text))

    def has_article(number):
        return bool(
            re.search(
                rf"\bartigo\s+{number}\b",
                normalized
            )
        )

    # Artigo 39: frequência e reprovação por faltas.
    mentions_absence = bool(
        question_words.intersection({
            "falta", "faltas", "frequencia",
            "ausencia", "ausencias", "ausentado"
        })
    )

    mentions_rejection = bool(
        question_words.intersection({
            "reprovado", "reprovada", "reprovacao"
        })
    )

    mentions_minimum = bool(
        question_words.intersection({
            "minima", "minimo", "precisa",
            "necessita", "manter"
        })
    )

    if (
        mentions_absence
        and mentions_rejection
        and has_article(39)
        and "25" in context_words
    ):
        if mentions_minimum:
            return (
                "O aluno precisa manter pelo menos 75% de "
                "frequência. Conforme o Artigo 39, a reprovação "
                "por falta ocorre quando o aluno ultrapassa "
                "25% de ausências."
            )

        return (
            "O aluno será reprovado por falta quando tiver "
            "se ausentado em mais de 25% das aulas, conforme "
            "o Artigo 39 do regulamento."
        )

    # Artigo 35: acesso à avaliação corrigida.
    if (
        "avaliacao" in question_words
        and "corrigida" in question_words
        and has_article(35)
        and "acesso" in context_words
    ):
        return (
            "Sim. É dever do professor permitir ao aluno "
            "o acesso ao instrumento de avaliação após a "
            "correção, conforme o Artigo 35, § 7º."
        )

    # Artigo 38: revisão da média final.
    if (
        "revisao" in question_words
        and "media" in question_words
        and has_article(38)
    ):
        if "professor" in question_words:
            return (
                "O professor tem igual período ao prazo "
                "concedido ao aluno para solicitar a revisão "
                "da média final, conforme o Artigo 38, "
                "parágrafo único."
            )

        return (
            "O aluno pode solicitar a revisão da média final "
            "em até 3 dias consecutivos após sua divulgação, "
            "conforme o Artigo 38, parágrafo único."
        )

    # Artigo 42: exercícios domiciliares.
    if (
        question_words.intersection({
            "compensar", "compensacao"
        })
        and question_words.intersection({
            "faltas", "ausencias"
        })
        and has_article(42)
    ):
        return (
            "Sim. O Artigo 42 prevê o regime de exercícios "
            "domiciliares, que permite a compensação de "
            "ausências quando atendidas as condições "
            "estabelecidas no regulamento."
        )

    # Artigo 43: prazo do exercício domiciliar.
    if (
        question_words.intersection({
            "prazo", "solicitar", "requerer"
        })
        and question_words.intersection({
            "domiciliar", "domiciliares", "exercicio"
        })
        and has_article(43)
        and "5" in context_words
        and "dias" in context_words
    ):
        return (
            "O requerimento do regime de exercício domiciliar "
            "deve ser protocolado em até 5 dias consecutivos, "
            "contados a partir da data do afastamento, "
            "conforme o Artigo 43."
        )

    # Artigo 40: hipóteses de abono de faltas.
    mentions_article_40_topic = bool(
        question_words.intersection({
            "abono", "militar", "judiciaria",
            "policial", "autoridade", "representacao",
            "estudantil", "colegiados", "paternidade",
            "licenca"
        })
    )

    if mentions_article_40_topic and has_article(40):

        if "militar" in question_words:
            return (
                "Sim. A convocação para cumprimento de "
                "serviços militares obrigatórios por lei "
                "é uma hipótese de abono de faltas, "
                "conforme o Artigo 40, inciso I."
            )

        if question_words.intersection({
            "judiciaria", "policial", "autoridade"
        }):
            return (
                "Sim. A convocação por autoridade judiciária "
                "ou policial está prevista para abono de faltas, "
                "conforme o Artigo 40, inciso II."
            )

        if question_words.intersection({
            "representacao", "estudantil", "colegiados"
        }):
            return (
                "Sim. O exercício de representação estudantil "
                "em órgãos colegiados, nos horários de reunião, "
                "está previsto para abono de faltas, conforme "
                "o Artigo 40, inciso III."
            )

        if question_words.intersection({
            "paternidade", "licenca"
        }):
            return (
                "Sim. O regulamento prevê licença-paternidade "
                "em caso de nascimento de filho, adoção ou "
                "guarda, conforme o Artigo 40, inciso IV."
            )

    # Artigo 26: rematrícula obrigatória e responsabilidade.
    mentions_rematricula = "rematricula" in question_words

    if mentions_rematricula and has_article(26):
        if question_words.intersection({
            "responsavel", "responsabilidade"
        }):
            return (
                "A rematrícula é de responsabilidade do aluno "
                "e ocorre pelo sistema acadêmico, conforme "
                "o Artigo 26, § 4º."
            )

        if question_words.intersection({
            "obrigatoria", "obrigatorio"
        }):
            return (
                "Sim. A rematrícula é obrigatória, de "
                "responsabilidade do aluno e realizada pelo "
                "sistema acadêmico, conforme o Artigo 26, § 4º."
            )

    # Artigo 32: hipóteses de cancelamento da matrícula.
    if has_article(32):

        # Inciso IX: não realizar rematrícula.
        if (
            mentions_rematricula
            and (
                "manutencao da vaga" in normalized
                or "rematricula" in context_words
            )
        ):
            return (
                "A matrícula pode ser cancelada se o aluno "
                "não realizar a rematrícula para manutenção "
                "da vaga, conforme o Artigo 32, inciso IX. "
                "A rematrícula também é obrigatória, conforme "
                "o Artigo 26, § 4º."
            )

        # Inciso I: ausências no início das aulas.
        if (
            "ingressante" in question_words
            and question_words.intersection({
                "faltas", "ausencias", "ausentar"
            })
            and "primeiros" in context_words
            and "14" in context_words
            and "7" in context_words
        ):
            return (
                "A matrícula do aluno ingressante pode ser "
                "cancelada quando ele se ausentar, sem "
                "justificativa, por 7 dias consecutivos ou não, "
                "nos primeiros 14 dias do início das aulas, "
                "conforme o Artigo 32, inciso I."
            )

        # Inciso II: reprovação no primeiro período.
        if (
            "ingressante" in question_words
            and question_words.intersection({
                "aprovado", "aprovada", "aprovacao"
            })
            and "primeiro" in question_words
            and "periodo" in question_words
            and "primeiro" in context_words
        ):
            return (
                "Sim. A matrícula do aluno ingressante pode "
                "ser cancelada se ele não for aprovado em "
                "ao menos um componente curricular no primeiro "
                "período letivo regular correspondente ao "
                "ingresso, conforme o Artigo 32, inciso II."
            )

        # Inciso III: percentual de progressão inferior a 33%.
        if (
            "progressao" in question_words
            and "33" in normalized
            and "33" in context_words
        ):
            return (
                "A matrícula pode ser cancelada quando, após "
                "o prazo mínimo de integralização previsto no "
                "PPC, o percentual de progressão do aluno for "
                "inferior a 33%, conforme o Artigo 32, inciso III."
            )

        # Inciso V: cancelamento solicitado pelo aluno.
        if (
            question_words.intersection({
                "solicitar", "solicita", "solicitado"
            })
            and question_words.intersection({
                "propria", "proprio", "aluno"
            })
            and "matricula" in question_words
            and "secretaria" in normalized
        ):
            return (
                "Sim. O aluno pode solicitar o cancelamento "
                "da própria matrícula mediante protocolo junto "
                "à Secretaria Acadêmica, dirigido ao "
                "Coordenador da Fatec, conforme o Artigo 32, "
                "inciso V."
            )

        # Inciso VII: expulsão em processo disciplinar.
        if (
            "expulsao" in question_words
            and "disciplinar" in question_words
            and "processo" in question_words
            and "expulsao" in context_words
        ):
            return (
                "Sim. A condenação à pena de expulsão em "
                "processo disciplinar é uma hipótese de "
                "cancelamento da matrícula, conforme o "
                "Artigo 32, inciso VII."
            )

        # Inciso VIII: veterano sem aprovação em dois períodos.
        if (
            "veterano" in question_words
            and question_words.intersection({
                "periodos", "periodo"
            })
            and question_words.intersection({
                "aprovacao", "aprovado", "obtiver"
            })
            and "veterano" in context_words
        ):
            return (
                "A matrícula do aluno veterano pode ser "
                "cancelada se ele não obtiver aprovação em "
                "nenhum componente curricular em dois períodos "
                "letivos, consecutivos ou não, respeitadas as "
                "exceções previstas no regulamento. Essa "
                "hipótese está no Artigo 32, inciso VIII."
            )

    return None


def hybrid_search(question, vectorstore, bm25, documents, k=6):
    expanded_question = expand_query(question)
    question_words = set(tokenize(question))

    semantic_results = vectorstore.similarity_search(
        question,
        k=8
    )

    expanded_results = vectorstore.similarity_search(
        expanded_question,
        k=8
    )

    def lexical_search(query, limit=8):
        scores = bm25.get_scores(tokenize(query))

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:limit]

        return [
            documents[i]
            for i in indices
            if scores[i] > 0
        ]

    lexical_results = lexical_search(expanded_question)

    rankings = [
        semantic_results,
        expanded_results,
        lexical_results,
    ]

    target_articles = set()

    if question_words.intersection({
        "falta", "faltas", "frequencia",
        "ausencia", "ausencias", "reprovado",
        "reprovada", "reprovacao"
    }):
        target_articles.add(39)

    if question_words.intersection({
        "abono", "militar", "judiciaria",
        "policial", "representacao", "colegiados",
        "paternidade", "estudantil", "licenca"
    }):
        target_articles.add(40)

    if question_words.intersection({
        "revisao", "media"
    }):
        target_articles.add(38)

    if question_words.intersection({
        "avaliacao", "corrigida"
    }):
        target_articles.add(35)

    if question_words.intersection({
        "domiciliar", "domiciliares", "exercicio"
    }):
        target_articles.update({42, 43})

    if "rematricula" in question_words:
        target_articles.update({26, 32})

    if question_words.intersection({
        "cancelamento", "cancelada", "cancelado",
        "cancelar", "ingressante", "veterano",
        "progressao", "expulsao"
    }):
        target_articles.add(32)

    if question_words.intersection({
        "trancamento", "trancar", "trancado"
    }):
        target_articles.add(31)

    if question_words.intersection({
        "desistencia", "desistir"
    }):
        target_articles.add(30)

    targeted_results = []

    for article_number in sorted(target_articles):
        pattern = re.compile(
            rf"\bArtigo\s+{article_number}\b",
            flags=re.IGNORECASE
        )

        for document in documents:
            if pattern.search(document.page_content):
                targeted_results.append(document)

    fused_scores = {}
    document_map = {}

    for ranking in rankings:
        for position, document in enumerate(ranking, start=1):
            key = (
                re.sub(
                    r"\s+",
                    " ",
                    document.page_content
                ).strip(),
                document.metadata.get("page")
            )

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + 1 / (60 + position)
            )

            document_map[key] = document

    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    ranked_results = [
        document_map[key]
        for key in ranked_keys
    ]

    combined = targeted_results + ranked_results

    unique_results = []
    seen = set()

    for document in combined:
        key = (
            re.sub(
                r"\s+",
                " ",
                document.page_content
            ).strip(),
            document.metadata.get("page")
        )

        if key not in seen:
            seen.add(key)
            unique_results.append(document)

    if target_articles:
        results = unique_results[:max(k, len(targeted_results))]
    else:
        results = unique_results[:k]

    return remove_obsolete_article_versions(results)


def prepare_context(results, documents=None):
    expanded_results = list(results)

    if documents:
        additional_results = []

        for document in results:
            articles = identify_articles(document.page_content)

            if "32" not in articles:
                continue

            document_index = next(
                (
                    i for i, item in enumerate(documents)
                    if (
                        item.page_content == document.page_content
                        and item.metadata.get("page")
                        == document.metadata.get("page")
                    )
                ),
                None
            )

            if document_index is None:
                continue

            for offset in range(1, 7):
                next_index = document_index + offset

                if next_index >= len(documents):
                    break

                next_document = documents[next_index]

                article_33 = re.search(
                    r"\bArtigo\s+33\b",
                    next_document.page_content,
                    flags=re.IGNORECASE
                )

                if article_33:
                    before_article_33 = (
                        next_document.page_content[:article_33.start()]
                    )

                    if before_article_33.strip():
                        additional_results.append(
                            Document(
                                page_content=before_article_33,
                                metadata=next_document.metadata
                            )
                        )
                    break

                additional_results.append(next_document)

        for document in additional_results:
            already_added = any(
                document.page_content == existing.page_content
                and document.metadata.get("page")
                == existing.metadata.get("page")
                for existing in expanded_results
            )

            if not already_added:
                expanded_results.append(document)

    context_parts = []

    for index, document in enumerate(expanded_results, 1):
        page = document.metadata.get("page")

        page_label = (
            page + 1 if isinstance(page, int) else page
        )

        articles = identify_articles(document.page_content)

        article_label = (
            ", ".join(f"Artigo {number}" for number in articles)
            if articles
            else "Continuação de trecho"
        )

        context_parts.append(
            f"[TRECHO {index} | PÁGINA {page_label} | "
            f"ARTIGOS: {article_label}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def create_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é um Assistente Acadêmico especializado
em interpretar regulamentos acadêmicos.

Responda sempre em português e utilize exclusivamente
as informações presentes no contexto.

REGRAS OBRIGATÓRIAS:

1. Leia todos os trechos antes de responder.
2. Responda diretamente ao que foi perguntado.
3. Combine informações de diferentes trechos quando necessário.
4. Não invente informações, regras, prazos ou artigos.
5. Identifique corretamente artigo, parágrafo e inciso.
6. Não confunda incisos com parágrafos.
7. Utilize somente redações atualizadas dos artigos.
8. Diferencie reprovação, aprovação e aproveitamento.
9. Não transforme a proibição de obter dispensa em
   proibição de solicitar aproveitamento.
10. Não contradiga a própria resposta.
11. Se não houver informação suficiente, informe isso claramente.
12. Não tente adivinhar.

Contexto:

{context}"""
        ),
        ("human", "{question}"),
    ])


def main():
    print("\n=== Assistente Acadêmico — RAG ===")
    print("Carregando banco de conhecimento...")

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
        print("Nenhum documento encontrado no banco vetorial.")
        return

    bm25 = BM25Okapi([
        tokenize(document.page_content)
        for document in documents
    ])

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    prompt = create_prompt()

    print(f"Documentos carregados: {len(documents)}")
    print("Sistema pronto! Digite 'sair' para encerrar.")

    while True:
        question = input("\nSua pergunta: ").strip()

        if question.lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break

        if not question:
            continue

        try:
            results = hybrid_search(
                question,
                vectorstore,
                bm25,
                documents,
                k=6
            )

            if not results:
                print("Nenhuma informação relevante encontrada.")
                continue

            print("\n--- TRECHOS RECUPERADOS ---")

            for index, document in enumerate(results, 1):
                page = document.metadata.get("page")

                page_label = (
                    page + 1 if isinstance(page, int) else page
                )

                print(f"\nTrecho {index} | Página {page_label}")
                print(document.page_content)

            context = prepare_context(results, documents)

            direct_answer = answer_specific_rule(
                question,
                results,
                context
            )

            print("\n--- RESPOSTA DO ASSISTENTE ---")

            if direct_answer is not None:
                print(direct_answer)
            else:
                response = (prompt | llm).invoke({
                    "context": context,
                    "question": question,
                })

                print(response.content)

            print("\n--- FONTES ---")

            pages = set()

            for document in results:
                page = document.metadata.get("page")

                if isinstance(page, int):
                    pages.add(page + 1)
                elif page is not None:
                    pages.add(page)

            for page in sorted(pages, key=str):
                print(f"- Página {page}")

        except Exception as error:
            print(f"Erro ao consultar o sistema: {error}")


if __name__ == "__main__":
    main()
