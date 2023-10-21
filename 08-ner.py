
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

sentence = '''The annual Tech Summit 2023 kicked off yesterday in Silicon Valley, drawing thousands of tech enthusiasts from around the world. The event, organized by the leading tech conglomerate InnovateX, promises to showcase the latest breakthroughs in artificial intelligence, blockchain, and renewable energy.

One of the highlights of the conference was a keynote speech by Dr. Emily Johnson, a renowned AI researcher from Stanford University. Dr. Johnson discussed the future of machine learning and its potential applications in healthcare. She also announced a collaboration between Stanford and InnovateX to develop AI-driven medical diagnostic tools.

In addition to Dr. Johnson's keynote, there were several panel discussions featuring industry experts. The panel on blockchain technology included speakers from major companies like Ethereum Corp and CryptoTech Solutions. They discussed the role of blockchain in financial services, supply chain management, and cybersecurity.

The renewable energy track at the summit featured presentations from leading companies in the field. SolarTech Inc unveiled its latest solar panel technology, which promises higher efficiency and lower costs. WindPower Innovations showcased their innovative wind turbine designs, aimed at making wind energy more accessible in urban areas.

Overall, Tech Summit 2023 is a gathering of some of the brightest minds in the tech world. It's a testament to the ongoing innovation and the role technology plays in shaping our future.
'''

doc = nlp(sentence)

for ent in doc.ents:
    print(ent.text, " - ", ent.label_)

displacy.serve(doc, style='ent')