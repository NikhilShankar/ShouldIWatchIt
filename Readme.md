https://github.com/egbertbouman/youtube-comment-downloader


run python ./setup/setup.py
select tensorflow cpu as the venv

activate the venv 
then run the folllowing pip

install 
```
pip install youtube-comment-downloader
pip install nltk
pip install seaborn
```


Use 

##### **1 DataCollection/DataCollection.ipynb**
-To get the comments into a csv file
-It will be saved inside datastore.

Once you download enough movies and comments

#### **2.** 
Run DataCombiner part of the same file. This will add all the comments into a single csv

#### **3.**
Now open Data cleaning and point the dataset_combined_path to point to the csv file you created in step 2
- Run all the codeblocks
- At the end it will generate another csv file inside cleaned folder.

#### **4.** 
- Open V1 folder and V1.ipynb
- Change the cleaned_csv_path to point to the file created at end of step 3
- Run all the code blocks. 
- Last code block you can paste a youtube link and it will predict the result.


