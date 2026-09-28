# Judge instructions

These are the exact instructions the judge received when the verdict matrices
in `data/matrices/` were collected. `collector.py` reads them through
`registry.json`.

| File | Criterion | Language | Matrices |
|---|---|---|---|
| `relevance_v2.txt` | relevance | Russian | GR1, LR2 |
| `auth_v3.txt` | authenticity | Russian | GA2, GA1, GA2_en_ru |
| `auth_eng_v3.txt` | authenticity | English | GA2_en |
| `coh_eng_v1.txt` | coherence | English | PDD |

`auth_eng_v3` is the English version of `auth_v3`, with the structure and the
order of the items preserved. GA2_en_ru was collected with the Russian
`auth_v3` on the English translation of the corpus. In the authors' collection
logs `auth_v3` was named `relevance_v3` (`former_name` in `registry.json`,
`instruction_logged` in `data/matrices/matrices.json`). The ids of the original
collections that appear in `results/` carry its short tag `v3`, e.g.
`REF_openai_local_Gemma4-26_v3_bidir`.

## How a prompt is built

The template is filled with Python's `str.format`:

- `{parent_A}`, `{reply_A}`, `{parent_B}`, `{reply_B}` receive the two
  "parent – reply" pairs, with runs of whitespace collapsed to single spaces.
- `{root_note}` receives a note for top-level comments, whose "parent" is the
  description of the video. The note says that the reply should be judged
  against the topic of the video rather than for literal correspondence with
  the description. It is added only when the collection runs with
  `root_clause=True`: all six corpus matrices do, P-DD does not. The note is
  also added only when at least one of the two pairs is a top-level comment.
  In that case the "both" variant from `registry.json` is used if both pairs
  are top-level; otherwise the "one" variant, with `{side}` = A or B. The note
  is in the language of the instruction. Otherwise `{root_note}` is empty.

The judge answers with one letter: A (pair A is better), B (pair B is better)
or E (about the same). The parser accepts the first standalone A, B or E; the
Cyrillic look-alikes А, В, Б, Е are mapped to Latin letters first. An answer
without a letter counts as E and is logged as unparsed.

## Reading aid: `relevance_v2` in English

This translation is for readers only; it was not used in any run. The English
version of `auth_v3` is `auth_eng_v3.txt`. The English root notes in
`registry.json` render the Russian ones.

```
Below are two parent comment — reply pairs.
Decide in which PAIR the reply is more relevant to its parent
comment (that is, it is a meaningful, on-topic and logical
continuation of its own parent in particular).

Pair A:
Parent A:
<<<
{parent_A}
>>>
Reply A:
<<<
{reply_A}
>>>

Pair B:
Parent B:
<<<
{parent_B}
>>>
Reply B:
<<<
{reply_B}
>>>

{root_note}Choose A or B ONLY if the difference in relevance is obvious.
In every other case choose E.

Specific cases where the correct answer is E:
- both replies clearly answer their parent
- both replies look like slogans or flaming unrelated to the topic
- the replies are very short (one or two words, interjections)
- the replies express similar emotions without a clear link to the content
- you cannot confidently tell one from the other

Signs that raise the relevance of one of the pairs: quoting the parent,
agreement or rebuttal on the substance, a clarification, direct address.
Signs of a bot reply or a planted message (they lower the relevance): slogans,
phrases unrelated to the parent's topic, general aggression without arguments.

Choose ONE option and return strictly a single letter with no explanation:
A — reply A is clearly more closely connected with its parent
B — reply B is clearly more closely connected with its parent
E — the relevance is comparable, or the difference is not obvious

Answer (one letter A, B or E):
```
