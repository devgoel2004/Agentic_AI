def build_prompt(query, results):
    # Here we are providing the results in which we join them.
    context = "\n\n".join(
        result['document']
        for result in results
    )
    prompt = f"""
    You are a helpful AI assistant.
    Answer the question using ONLY the information provided
    in the context.

    If the answer is not available in the context, say:
    "I don't have the information based on the provided context."

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    return prompt
