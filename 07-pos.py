import nltk
from nltk import word_tokenize

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = start_p[state] * emit_p[state].get(obs[0], 0)
        path[state] = [state]

    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, possible_state) = max([(V[t-1][prev_state] * trans_p[prev_state].get(state, 0) * emit_p[state].get(obs[t], 0), prev_state)
                                          for prev_state in states if V[t-1][prev_state] > 0], default=(0.0, None))
            V[t][state] = prob
            new_path[state] = path[possible_state] + [state]

        path = new_path

    n = len(obs)
    (prob, state) = max((V[n-1][state], state) for state in states)

    return path[state]

states = ['Noun', 'Verb', 'Adjective', 'Adverb']

text = '''I love programming in Python.'''
observations = word_tokenize(text)

initial_probabilities = {'Noun': 0.3, 'Verb': 0.3, 'Adjective': 0.2, 'Adverb': 0.2}

transition_matrix = {
    'Noun': {'Noun': 0.3, 'Verb': 0.4, 'Adjective': 0.2, 'Adverb': 0.1},
    'Verb': {'Noun': 0.1, 'Verb': 0.4, 'Adjective': 0.3, 'Adverb': 0.2},
    'Adjective': {'Noun': 0.2, 'Verb': 0.3, 'Adjective': 0.4, 'Adverb': 0.1},
    'Adverb': {'Noun': 0.1, 'Verb': 0.2, 'Adjective': 0.2, 'Adverb': 0.5},
}

emission_matrix = {
    'Noun': {'I': 0.1, 'love': 0.05, 'programming': 0.1, 'in': 0.2, 'Python': 0.05},
    'Verb': {'I': 0.05, 'love': 0.2, 'programming': 0.1, 'in': 0.05, 'Python': 0.1},
    'Adjective': {'I': 0.05, 'love': 0.1, 'programming': 0.15, 'in': 0.05, 'Python': 0.05},
    'Adverb': {'I': 0.05, 'love': 0.05, 'programming': 0.05, 'in': 0.1, 'Python': 0.15}
}

result = viterbi(observations, states, initial_probabilities, transition_matrix, emission_matrix)

print("Sentence: I love programming in Python.")
print("Most likely sequence of POS tags:", result)
