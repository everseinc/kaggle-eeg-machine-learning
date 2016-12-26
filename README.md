# Tensorflowを用いたEEGデータの機械学習

- [Kaggle EEGデータセットの用意](#eeg-data)

- [TensorFlowインストール手順](#tensorflow-install)

<a name="eeg-data"></a>

## Kaggle EEGデータセットの用意
 - **このデータを用意しないとエラーが出ます。**
 - **また、このデータセットは非常にサイズが大きいので注意してください。**
 - **この一連の作業はいづれ自動化します**
 
 ### データセットのダウンロード
 
 以下のURLから`train`及び`test`用のデータセットをダウンロードし、`EEG_grasp_and_left_data`という名前のフォルダに入れてください。  
 https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data  
 
 ### フォルダ構成
 
> Cnn/  
>  ├ tensorflow352/  
>  │　└ csv_manager.py/  
>  │　└ eeg_cnn.py/  
>  │　└ helper.py/  
>  │　└ ...  
>  └ EEG_grasp_and_left_data/  
>　 　└ train/  
>　 　　　└ subj1_series1_data.csv  
>　 　　　└ subj1_series1_events.csv  
>　 　　　└ subj1_series2_data.csv  
>　 　　　└ ...  
>　 　└ test/  
>　 　　　└ subj1_series9_data.csv  
>　 　　　└ subj1_series10_data.csv  
>　 　　　└ ...  
 
<a name="tensorflow-install"></a>

## TensorFlowインストール手順
- **【推奨】** CentOS 7.0

### 必要なパッケージのインストール
```bash
sudo yum update -y
sudo yum groupinstall "Development Tools”
sudo yum install -y zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel
```

### python3.5.2をインストール
```bash
sudo wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar -xf Python-3.5.2.tgz
cd Python-3.5.2
./configure --with-threads
sudo make altinstall
cd
```

### pipのインストール
```bash
sudo wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip -V
pip install --upgrade pip
```

### pyenv と pyenv-virtualenv をインストール
```bash
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
git clone git://github.com/yyuu/pyenv-virtualenv.git ./.pyenv/plugins/pyenv-virtualenv
git clone https://github.com/yyuu/pyenv-virtualenvwrapper.git ./.pyenv/plugins/pyenv-virtualenvwrapper

vi ~/.bashrc

# ~/.bashrc に下記を末尾に追加
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)”
eval "$(pyenv virtualenv-init -)”

source ~/.bashrc
echo $PATH

cd ~/.pyenv
git pull origin
cd

pyenv versions
```

### Python3.5系のインストール
```bash
pyenv install 3.5.2
pyenv rehash
pyenv versions
```

### TensorFlowのpython環境を分ける
```bash
cd /vagrant/tensorflow352
pyenv virtualenv 3.5.2 tensorflow352
pyenv rehash
pyenv local tensorflow352
pyenv versions
```

### Tensorflowのインストール
```bash
pip install --upgrade pip
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
```
