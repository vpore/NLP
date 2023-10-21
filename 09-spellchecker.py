import contextualSpellCheck
import spacy

nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
# doc = nlp('Spell Checker is a sekuence-to-sequnce pipline that detects and corrects speling errors in yor input tsxt.')

doc = nlp('A spell-cheker is a tul used for analyzing and validating speling mistakes in the tekst.')
print(len(doc._.suggestions_spellCheck))

print(doc._.suggestions_spellCheck)

print(doc._.outcome_spellCheck)
