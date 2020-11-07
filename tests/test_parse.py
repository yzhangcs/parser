# -*- coding: utf-8 -*-

import nltk
import supar
from supar import Parser


def test_parse():
    sentence = ['The', 'dog', 'chases', 'the', 'cat', '.']
    for name in supar.PRETRAINED:
        parser = Parser.load(name)
        parser.predict([sentence], prob=True)


def test_bert():
    nltk.download('punkt')
    sentence = nltk.word_tokenize('''
        No, it wasn't Black Monday.
        But while the New York Stock Exchange didn't fall apart Friday as the Dow Jones Industrial Average
        plunged 190.58 points - most of it in the final hour - it barely managed to stay this side of chaos.
        Some "circuit breakers" installed after the October 1987 crash failed their first test, traders say,
        unable to cool the selling panic in both stocks and futures.
        The 49 stock specialist firms on the Big Board floor - the buyers and sellers of last resort
        who were criticized after the 1987 crash - once again couldn't handle the selling pressure.
        Big investment banks refused to step up to the plate to support the beleaguered floor traders
        by buying big blocks of stock, traders say.
        Heavy selling of Standard & Poor's 500-stock index futures in Chicago relentlessly beat stocks downward.
        Seven Big Board stocks - UAL, AMR, BankAmerica, Walt Disney, Capital Cities/ABC,
        Philip Morris and Pacific Telesis Group - stopped trading and never resumed.
        The finger-pointing has already begun. "The equity market was illiquid.
        Once again {the specialists} were not able to handle the imbalances on the floor of the New York Stock Exchange,"
        said Christopher Pedersen, senior vice president at Twenty-First Securities Corp.
        Countered James Maguire, chairman of specialists Henderson Brothers Inc.:
        "It is easy to say the specialist isn't doing his job.
        When the dollar is in a free-fall, even central banks can't stop it.
        Speculators are calling for a degree of liquidity that is not there in the market."
        Many money managers and some traders had already left their offices early Friday afternoon on a warm autumn day -
        because the stock market was so quiet.
        Then in a lightning plunge,
        the Dow Jones industrials in barely an hour surrendered about a third of their gains this year,
        chalking up a 190.58-point, or 6.9%, loss on the day in gargantuan trading volume.
        Final-hour trading accelerated to 108.1 million shares, a record for the Big Board.
        At the end of the day, 251.2 million shares were traded.
        The Dow Jones industrials closed at 2569.26.
        The Dow's decline was second in point terms only to the 508-point Black Monday crash that occurred Oct. 19, 1987.
        ''')
    parser = Parser.load('biaffine-dep-bert-en')
    parser.predict([sentence], prob=True)
