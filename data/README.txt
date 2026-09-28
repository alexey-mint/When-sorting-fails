YouTube comment dialogues: 399 parent comment – reply pairs from a
socio-political discussion, translated into English

Version:  1.0
Creator:  Oleksii Mints
License:  Creative Commons Attribution-NonCommercial 4.0 International
          (CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/


FILES

  corpus_400-2_en.csv   the data set: 399 rows, 4 columns
  README.txt            this file


DESCRIPTION

The data set contains 399 "parent comment – reply" pairs from the comments
under one YouTube video on a socio-political topic, translated into English.
The original comments are predominantly in Russian.

Sampling. The pairs are an enriched subsample of a corpus of about 3,700
pairs. A random sample, distributed uniformly over the publication time of
the comments, was enriched with manually selected extreme cases: clearly
templated or coordinated replies and clearly live dialogue. Part of the
average examples was then removed to keep the size. The row order is
therefore not random with respect to how templated or live a reply is.

Parent. The parent of a reply is determined as follows:
  - for a top-level comment, it is the description of the video;
  - for a reply that does not start with an @-mention, it is the root
    comment of the thread;
  - for a reply that starts with an @-mention, it is the comment of the
    mentioned user, found by walking back through the thread. If that user
    is not in the thread, the root comment of the thread is used instead.
Several replies can share a parent; the file holds 240 distinct parents.

Translation. The texts were translated by the GPT6-Astra model. The model
was instructed to render the meaning rather than the letter, including the
deliberate spelling distortions, character substitutions and allusions with
which the authors evade moderation.

Anonymization. In line with the YouTube data policy, the original comments
are not published. The file contains no user names, pseudonyms, comment
identifiers, links, dates or like counts.


COLUMNS

  id                   row number, 0-398
  parent_text_clean    the parent: the video description, the root comment
                       of the thread or the comment of the mentioned user
  textOriginal_clean   the reply
  _parent_method       how the parent was determined:
                         top_level          top-level comment (112 rows)
                         second_level       reply without an @-mention (77)
                         mention_backtrack  reply with an @-mention (210)

Texts have a median length of 36 words for parents and 15 words for
replies.

File format:
  - UTF-8 without BOM, comma-separated, with a header row;
  - records end with CRLF, and fields are quoted where needed;
  - 73 texts contain line breaks and 55 replies contain emoji, so read the
    file with a CSV parser rather than line by line.

With pandas:

  import pandas as pd
  df = pd.read_csv("corpus_400-2_en.csv", keep_default_na=False)


TERMS OF USE

The comments were publicly posted on YouTube. Copyright in the original
comments remains with their authors; the license of this data set covers
the selection, cleaning and translation of the texts. Please do not attempt
to identify the authors of the comments.
