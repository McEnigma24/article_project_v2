#!/bin/bash

install_hook()
{
    DIR_SCRIPTS="scripts"
    PATH_MY_HOOK="$DIR_SCRIPTS/formating_hook.sh"
    PATH_GIT_HOOK=".git/hooks/pre-commit"

    cp $PATH_MY_HOOK $PATH_GIT_HOOK && chmod +x $PATH_GIT_HOOK
    chmod +x $PATH_GIT_HOOK
}

user_name="McEnigma24"

repo_name="$(basename "`pwd`")"
echo "Do you want repo name as: $repo_name ?    enter to approve / 'n' to choose"
read choice

if [ "$choice" != "" ]; then
{
    echo "input name: "
    read repo_name

    if [ "$repo_name" == "" ]; then
    {
        echo "invalid repo name"
        exit
    }
    fi
}
fi

sudo apt install gh
gh auth login

git init
git add .
git commit -m "Init"

echo "private = enter"
echo "public = 'n'"
read choice
if [ "$choice" == "" ]; then
{
    gh repo create $repo_name --private
}
else
{
    gh repo create $repo_name --public
}
fi

git remote add origin https://github.com/$user_name/$repo_name.git
git remote set-url origin git@github.com:$user_name/$repo_name.git

git branch -M main
git push -u origin main

install_hook # my addition

rm -f "$0"
./push.sh