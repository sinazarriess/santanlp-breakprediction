
## Projekt: Finetuning BERT's next-sentence prediction for narrative level detection
## @Judith

Was gibt es schon?

- ein Paper, under construction:
    * ... Link kommt separat

- Hintergrundliteratur zu ähnlichen Tasks, also Segmentierung von längeren Texten
    * Chapter Captor: Text Segmentation in Novels, https://aclanthology.org/2020.emnlp-main.672/
    * Detecting Scenes in Fiction: A new Segmentation Task, https://aclanthology.org/2021.eacl-main.276/

- Daten zum Evaluieren und Rumspielen mit narrativen Ebenen: https://github.com/nilsreiter/santanlp-corpus
    * corpus1 enthält lit. Texte, die zufällig geshuffelt wurden, die "<BREAK>"-tags markieren die Textgrenzen
    * corpus2-4 enthält gekürzte Originaltexte ohne breaks (für die wir aber Annotationen haben)

- Skripte zum Testen der next sentence prediction auf dem santanlp-corpus:
    * https://github.com/sinazarriess/santanlp-breakprediction/
    * hier am besten bei "try_bert.ipynb" anfangen

- erste Ergebnisse:
    * die off-the-shelf next sentence prediction von BERT scheint nicht so gut für lit. Texte zu funkionieren, er erkennt schon ein paar breaks in corpus1, aber längst nicht alle
    * die NSP funktioniert noch weniger gut für die echten narrativen Ebenen corpus2-4

Was wäre gut zu haben?

- die Idee wäre, die NSP von BERT auf corpus1/train finezutunen und erstmal zu gucken, ob die predictions auf corpus1/test besser werden
   * ich hab angefangen, das zu programmieren: https://github.com/sinazarriess/santanlp-breakprediction/blob/main/train_par_pairs.py
   * ist aber noch nicht getestet...

- das, was uns eigentlich interessiert, wäre, ob wir mit dem finetuning auf corpus1 auch bessere predictions auf corpus2-4 bekommen

- die richtigen annotierten Daten zu narrativen Ebenen sind noch in einem etwas komplizierten Format:
    * https://github.com/SharedTasksInTheDH/phase-1-round-2-test-corpus, (SANTA1-7)
    * ein Tool, mit dem man sich die Daten anschauen kann: https://github.com/nilsreiter/CorefAnnotator/
 ... das brauchst du wahrscheinlich erstmal nicht
