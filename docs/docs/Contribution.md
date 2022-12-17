

## Want to contribute?
 Great!!! You all are welcome. 

If you are a beginner then it is highly recommended that you go through <a href=a"https://github.com/gabrieldemarmiesse/getting_started_open_source">Getting started with open source guidelines </a>.

1. Fork this repository into your GitHub account.

2. Then clone the repository to your local machine using `git clone` command. Make sure you enter your user name. 
OR after forking you can simply press the <span style='color:white;background-color:green;border-radius:2px;padding:5px' ><> Code</span> button and copy the URL, open terminal and paste the url like this `git clone <the_copied_URL>`
```
git clone https://github.com/you_user_name/AutoNN.git
```

3. After step `2.` Create a new branch using `git checkout` command. And then install all the dependencies. 
```
git checkout -b branch-name-here

pip install -r requirements.txt
```

4. After this you can go ahead, make some changes, and add some missing funtionality if you like. 

5. Commit your changes using the `git commit` command

Example:
```
git commit -m "your message goes here"
```
6. Push the changes to the remote repo using
``` 
git push origin <branch-name>
```
7. Submit a pull request to the upstream repository with a proper message. If you are trying to fix an issue then the type of issue tag must be present. 

## Docs Contribution. 

install Mk Docs
```
pip install mkdocs-material
```

You can check the documentation <a href="https://squidfunk.github.io/mkdocs-material/">here</a>.

Goto the `../AutoNN/docs/` folder in your local repo.
!!! Folder_Structure
    ```
    docs/
        docs/
        mkdocs.yml
    ```
Use this command to see the dummy website at you local browser `mkdocs serve`. Whatever changes you make on documentation will be rendered in runtime. 






<!-- 
Now to test the GUI you can locally install the package using the following command.
```
pip install .
```
!!! attention 
    Make sure you are in the same directory where the `setup.py` file is -->

<!-- Open your terminal and copy paste the following command to start the GUI
```
autonn
```
This should work and you can locally check whether the GUI is working as expected.

!!! Note
    The GUI was built using tkinter and `ttkbootstrap==0.5.1`.  
     -->