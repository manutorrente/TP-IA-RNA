import pyparsing as pp
from collections import Counter

# mock_string_2 = '''
# (Chinesische) Zwergwachtel {f}	Chinese painted quail [Excalfactoria chinensis, syn.: Coturnix chinensis, Synoicus chinensis]	noun	[orn.] [T] 
# (Chinesische) Zwergwachtel {f}	king quail [Excalfactoria chinensis, syn.: Coturnix chinensis, Synoicus chinensis]	noun	[orn.] [T] 
# (Chinesischer) Blauglockenbaum {m}	foxglove tree [Paulownia tomentosa]	noun	[bot.] [T] 
# (Chinesischer) Kundekäfer {m}	adzuki / azuki bean weevil [Callosobruchus chinensis, syn.: Bruchus chinensis]	noun	[entom.] [T] 
# (Chinesischer) Kundekäfer {m}	Chinese bruchid [Callos
# '''
# #should be {Zwerwachtel: f, Blauglockenbaum: m, Kundekäfer: m}


# mock_string_1 = '''
# (das) Eis von etw. [Dat.] entfernen	to de-ice sth.	verb	
# (das) Eiweiß zu Schnee schlagen	to whisk the egg white / whites till stiff	verb	[gastr.] 
# (das) Elend überwinden	to overcome adversity	verb	
# (das) Forellenquintett [Franz Schubert]	(the) Trout Quintet		[F] [mus.] 
# (das) Fremde {n}	(the) alien	noun	
# (das) Fremde {n}	(th
# '''
# #should be {Eis: n, Eiweiß: n, Elend: n, Forellenquintett: n, Fremde: n}



de_chars = "äöüÄÖÜß"

de_alphas = pp.alphas + de_chars


#wrapper for the ParserElement
class DeNounParser: 
    def __init__(self, parser, converter, word_position = 0, gen_position= 1) -> None:
        self.word_position: int = word_position
        self.gender_position: int = gen_position
        self.converter = converter
        self.parser:pp.ParserElement = parser

    def convert(self, value):
        return self.converter[value]

    def scan_text(self, text) -> list[tuple[str, str]]:
        scanned = self.parser.scan_string(text)
        return list(map(lambda x: self.process_single_parse_result(x[0]), scanned))


    def process_single_parse_result(self, result: pp.ParseResults) -> tuple[str, str]: #(word, gender)
        result = result.as_list()
        word = result[self.word_position] + result[self.word_position + 1]
        gender = self.converter[result[self.gender_position]]
        return (word, gender)



class TextParse:
    def __init__(self, parsers: list[DeNounParser]) -> None:
        self.parsers = parsers
        self.dictionary = {}


    def parse(self, text):
        new_nouns = []
        for parser in self.parsers:
            new_nouns += parser.scan_text(text)

        for noun in new_nouns:
            self.add_de_nouns(noun[0], noun[1])


    def add_de_nouns(self, key, value):
        if key not in self.dictionary:
            self.dictionary[key] = Counter(value)
        else:
            self.dictionary[key].update(value)

    def empty_dictionary(self):
        self.dictionary = {}

capital_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"

german_noun = pp.Char(capital_letters) + pp.Word(de_alphas)

raw_parser_1  = (pp.CaselessLiteral("(der)") | pp.CaselessLiteral("(die)") | pp.CaselessLiteral("(das)")) + german_noun

raw_parser_2 = german_noun + (pp.CaselessLiteral(f"{{n}}")| pp.CaselessLiteral(f"{{f}}") | pp.CaselessLiteral(f"{{m}}"))

raw_parser_3 = (pp.CaselessLiteral("der") | pp.CaselessLiteral("die") | pp.CaselessLiteral("das")) + german_noun

de_noun_parser_1 = DeNounParser(raw_parser_1, {"(der)": "m", "(die)": "f", "(das)": "n"}, word_position=1, gen_position=0)
de_noun_parser_2 = DeNounParser(raw_parser_2, {"{n}": "n", "{f}": "f", "{m}": "m", "{pl}": "p"}, word_position=0, gen_position=2)
de_noun_parser_3 = DeNounParser(raw_parser_3, {"der": "m", "die": "f", "das": "n"}, word_position=1, gen_position=0)

text_parser = TextParse([de_noun_parser_2])
