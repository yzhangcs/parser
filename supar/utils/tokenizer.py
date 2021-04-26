# -*- coding: utf-8 -*-


class Tokenizer:

    def __init__(self, lang='en'):
        import stanza
        try:
            self.pipeline = stanza.Pipeline(lang=lang, processors='tokenize', verbose=False, tokenize_no_ssplit=True)
        except Exception:
            stanza.download(lang=lang, resources_url='stanford')
            self.pipeline = stanza.Pipeline(lang=lang, processors='tokenize', verbose=False, tokenize_no_ssplit=True)

    def __call__(self, text):
        return [i.text for i in self.pipeline(text).sentences[0].tokens]
