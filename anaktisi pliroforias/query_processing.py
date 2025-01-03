import re

# Δημιουργία κλάσης για την επεξεργασία των ερωτημάτων
class QueryProcessor:
    def __init__(self, data):                           # Λήψη και αποθήκευση των δεδομένων
        self.data = data

    def tokenize(self, query):                          # Χωρισμός των λέξεων του ερωτήματος
        return re.findall(r'\w+', query.lower())

    def search(self, query):                            # Αναζήτηση των δεδομένων
        tokens = self.tokenize(query)
        results = set(self.data)
        
        if 'and' in tokens:                             # Έλεγχος αν το ερώτημα περιέχει τη λέξη 'and' και επιστροφή των αποτελεσμάτων
            and_index = tokens.index('and')
            left_tokens = tokens[:and_index]
            right_tokens = tokens[and_index+1:]
            results = self._and_search(left_tokens, right_tokens)
        elif 'or' in tokens:                            # Έλεγχος αν το ερώτημα περιέχει τη λέξη 'or' και επιστροφή των αποτελεσμάτων
            or_index = tokens.index('or')
            left_tokens = tokens[:or_index]
            right_tokens = tokens[or_index+1:]
            results = self._or_search(left_tokens, right_tokens)
        elif 'not' in tokens:                           # Έλεγχος αν το ερώτημα περιέχει τη λέξη 'not' και επιστροφή των αποτελεσμάτων
            not_index = tokens.index('not')
            left_tokens = tokens[:not_index]
            right_tokens = tokens[not_index+1:]
            results = self._not_search(left_tokens, right_tokens)
        else:                                           # Επιστροφή των αποτελεσμάτων
            results = self._simple_search(tokens)
        
        return results

    def _simple_search(self, tokens):                   # Απλή αναζήτηση
        return {item for item in self.data if all(token in item.lower() for token in tokens)}

    def _and_search(self, left_tokens, right_tokens):       # Αναζήτηση με τη χρήση του 'and'
        left_results = self._simple_search(left_tokens)
        right_results = self._simple_search(right_tokens)
        return left_results & right_results

    def _or_search(self, left_tokens, right_tokens):        # Αναζήτηση με τη χρήση του 'or'
        left_results = self._simple_search(left_tokens)
        right_results = self._simple_search(right_tokens)
        return left_results | right_results

    def _not_search(self, left_tokens, right_tokens):       # Αναζήτηση με τη χρήση του 'not'
        left_results = self._simple_search(left_tokens)
        right_results = self._simple_search(right_tokens)
        return left_results - right_results

