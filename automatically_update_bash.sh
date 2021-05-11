wget https://raw.githubusercontent.com/KazegamiKuon/Deep-learning-and-Machine-Learning-Workflow/master/automatically_destroy_environment.sh
wget https://raw.githubusercontent.com/KazegamiKuon/Deep-learning-and-Machine-Learning-Workflow/master/automatically_initialize_environment.sh
if [ ! -d "./git-actions-practice/" ]; then
    git clone https://github.com/KazegamiKuon/git-actions-practice.git    
fi
cd ./git-actions-practice/
git pull
cd ../