# MelodyGAN

How to use mp3_converter.py (requires pydub):

If first argument is "-c" then convert an entire directory (second argument) of .mp3 into .npy.
If first argument is "-g" then generate an .mp3 from a .npy (second argument).
Else fail.

How to use midi to mp3 converter (requires timidity):

Will convert all .midi in /midi and output a .mp3 in /mp3 for each file.

./tomp3.sh


How to use extract_images:
python extract.py filepath_to_a_directory_with_image(from_kaggle) style_file
style_file = impressionism.txt or abstract.txt
