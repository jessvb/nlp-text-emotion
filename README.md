# NLP Text Emotion Recognition
*Detect emotion sentence-by-sentence in local text files!* This repository was mirrored from [lukasgarbas/nlp-text-emotion](https://github.com/lukasgarbas/nlp-text-emotion). (Note that a fork would have been preferable to a mirror, but [LFS does not support pushing objects to public forks](https://github.com/git-lfs/git-lfs/issues/1906), so a mirror had to do.)

## Usage
1. Clone the project locally
2. In the project root, create an `input` folder and an `output` folder
3. Place `.txt` file(s) containing sentences/paragraphs (that you'd like to use emotion recognition on) into the `input` directory
    - Note: Each sentence should be delimited by `.` or `!` (as in regular English)
4. In a terminal, `cd` into the project root
5. Run `pip install -r requirements.txt`
6. Run `python bert_predict.py` to use the pre-trained BERT NLP model to recognize emotions in each sentence
7. View the output `.csv` files with the recognized emotions (per sentence) in the `output` directory

## Train your own model
If you'd like to train your own model, see the `bert_init.py` file.