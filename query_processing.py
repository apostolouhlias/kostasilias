import re

class QueryProcessor:
    def __init__(self, data):
        self.data = data

    def tokenize(self, query):
        return re.findall(r'\w+', query.lower())

    def search(self, query):
        tokens = self.tokenize(query)
        results = set(self.data)
        
        if 'and' in tokens:
            and_index = tokens.index('and')
            left_tokens = tokens[:and_index]
            right_tokens = tokens[and_index+1:]
            results = self._and_search(left_tokens, right_tokens)
        elif 'or' in tokens:
            or_index = tokens.index('or')
            left_tokens = tokens[:or_index]
            right_tokens = tokens[or_index+1:]
            results = self._or_search(left_tokens, right_tokens)
        elif 'not' in tokens:
            not_index = tokens.index('not')
            left_tokens = tokens[:not_index]
            right_tokens = tokens[not_index+1:]
            results = self._not_search(left_tokens, right_tokens)
        else:
            results = self._simple_search(tokens)
        
        return results

    def _simple_search(self, tokens):
        return {item for item in self.data if all(token in item.lower() for token in tokens)}

    def _and_search(self, left_tokens, right_tokens):
        left_results = self._simple_search(left_tokens)
        right_results = self._simple_search(right_tokens)
        return left_results & right_results

    def _or_search(self, left_tokens, right_tokens):
        left_results = self._simple_search(left_tokens)
        right_results = self._simple_search(right_tokens)
        return left_results | right_results

    def _not_search(self, left_tokens, right_tokens):
        left_results = self._simple_search(left_tokens)
        right_results = self._simple_search(right_tokens)
        return left_results - right_results

# Example usage:
if __name__ == "__main__":
    data = ["The quick brown fox", "jumps over the lazy dog", "Python programming is fun", "Hello world"]
    qp = QueryProcessor(data)
    
    print(qp.search("quick and fox"))
    print(qp.search("quick or lazy"))
    print(qp.search("quick not lazy"))
    print(qp.search("Python"))