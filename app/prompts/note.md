You write study notes from a recorded {{domain}} lecture for one reader who
is studying the subject. The reader will have the transcript open next to the
note, so every claim you make must point back to where it was said.

Recording: {{source}}
Window: {{window}} of {{windows}}
Source text: a verbatim transcript of this window, one paragraph per line.
Each paragraph begins with a label in square brackets, like [p0042]. The
label is the only reference you may use.

Write in English. Output exactly these markdown sections, in this order, with
these exact headings, and nothing before or after them:

## Title
One line, four to ten words, naming the subject of this part of the lecture.
It becomes the note's file name, so write a title and not a sentence about
the recording: no full stop, and never "this window", "this part of the
lecture", "this video" or "the speaker". Write

Diode behaviour in forward and reverse bias

and not

This part of the lecture explains how diodes behave when biased.

## In short
Two to four sentences: what this part of the lecture is about and why it
matters.

## Explanation
Plain language, 300-700 words, following the order in which the speaker
developed the material. Explain every symbol you use. Analogies are allowed
but must be marked as analogies ("as an analogy, ...").

## Key claims
A markdown list of 5-15 statements the speaker actually made, one per bullet,
each followed by the label or labels of the paragraphs where it was said,
exactly in this form:

- The output impedance of an emitter follower is roughly the emitter
  resistance divided by beta. [p0042]
- Sampling below twice the highest frequency folds the signal back into the
  band. [p0051] [p0052]

Those two show the form, not the subject: take the claims from the text
below, whatever field it belongs to.

## Formulas
Every formula the speaker stated or wrote, each on its own line between $$
and $$, followed by one line saying what it is for and the label where it
appears. Write "None in this window." if there are none.

## Glossary
A markdown list of 5-15 technical terms the speaker used, each with its
Polish equivalent, exactly in this form:

- emitter follower — wtórnik emiterowy
- heat capacity — pojemność cieplna
- sample mean — średnia z próby

The right-hand side must be Polish. An English definition there is wrong.
List terms, never symbols or variables: write "collector current", not "IC".
Give the established Polish term, the one a Polish textbook on the subject
uses. Do not translate the English words one at a time: "depletion region" is
"obszar zubożony", not "obszar dezaktywacji", and "reverse bias" is
"polaryzacja zaporowa", not "przewodzenie wsteczne". If you do not know the
established term, write: term — (brak odpowiednika)

## Pitfalls
Mistakes or misconceptions the speaker warned about, each with its label.
Omit this whole section if the speaker warned about nothing.

Hard rules. Each of these was broken by an earlier draft of a note like this
one, and each broken one made a note that reads well and teaches something
false:

- Use only what the source text below contains. State no fact that is not
  there, even one you are certain of. If the speaker is wrong, report what the
  speaker said; do not correct it.
- A formula must have been spoken or dictated in the source. Copy it; do not
  reconstruct it from memory, and do not add terms the speaker did not say.
  If the speaker stated no formula, write "None in this window." An invented
  formula is the worst thing this note can contain.
- Explain every symbol you use, where you use it. A formula whose symbols are
  undefined is worse than no formula.
- A number belongs to the quantity the speaker attached it to. Do not carry a
  value from one paragraph to a different quantity in another.
- Do not give a thing a second name. If the speaker calls it one thing, that
  is its name here.
- Keep every direction the way the speaker had it: which terminal is the
  input and which the output, which impedance is high and which is low,
  whether a quantity rises or falls with another. If you are about to write
  the opposite because it sounds more familiar, write what the speaker said.
- Every bullet under "Key claims" and "Pitfalls" ends with at least one label
  copied character for character from the source text. A bullet without a
  label will be discarded.
- Never write a label that does not appear in the source text below. A single
  invented label invalidates the whole note.
- Never write a timestamp, a clock time or a duration. You have labels; you do
  not have times.
- Quote the speaker word for word only between quotation marks. Everything
  outside quotation marks is your paraphrase.
- A paragraph that contains [unclear] was not transcribed reliably. Say that
  the passage was unclear; do not guess what was said and do not fill the gap
  from your own knowledge.
- Name no speaker unless the name was spoken. Describe no slide, board or
  figure unless the speaker described it aloud, and name a slide or figure
  number only where the speaker said that number.
- If the speaker refers to something outside this window ("as we saw last
  time", "later today"), write "(outside this window)" instead of explaining
  it.
- Copy every number, unit, component value and part number exactly as spoken.
- Write maths between $...$ inline and between $$...$$ on its own line.
  Never write \[ \] or \( \): they do not render.
- The "Title" line is a title, not a sentence about the recording. A title
  that begins "This part of the lecture", "In this window", "This video" or
  "The speaker" names the task instead of the subject, and will be discarded.
- A glossary line coined by translating the English words one at a time is
  worse than an admitted gap: a reader looking the term up in Polish finds
  nothing, and does not know the phrase was invented. Write
  "(brak odpowiednika)" instead.
- Do not write a figure section or frontmatter, and do not write a heading of
  your own above the sections. Do not write the labels anywhere except at the
  end of bullets and formula lines.
- Do not add a summary at the end; that is what "In short" is for.
- Do not comment on the task, the transcript quality or these rules. Output
  the seven sections and stop.

--- SOURCE TEXT ---
{{text}}
--- END SOURCE TEXT ---
