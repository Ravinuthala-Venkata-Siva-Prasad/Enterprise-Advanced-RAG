class InputGuard:

    def __init__(self, max_length=2000):
        self.max_length = max_length

    def validate(self, query):

        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        if len(query) > self.max_length:
            raise ValueError("Query exceeds maximum length")

        if "\x00" in query:
            raise ValueError("Query contains invalid characters")

        return query