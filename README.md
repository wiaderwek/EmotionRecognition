###Generowanie ścieżek dźwiękowych do flimów na podstawie emocji

### Celem projektu jest tworzenie muzyki na podstawie emocji zawartych w filmach

### Opis zawartości bieżącego folderu

### video_analysis
Folder zawierający logi z uczenia sieci odpowiedzialnej za analizowanie filmów i folder z nauczonym modelem sieci.
Logi z uczenia: video_analysis/out/logs/
Model nauczonej sieci: video_analysis/out/models/

### music_generation
Folder zawierający plik nauczonego modelu sieci, jak również wygenerowane przez sieć fragmenty utowrów muzycznych.
Logi z uczenia: music_generation/out/logs/
Wygenerowane fragmenty utworów: music_generation/out/samples/

### examples
Folder zamierający przykładowe filmy do testowania sieci i przykłady wygenerowanych plików muzycznych.
Filmy do testowania: examples/films/
Przykłady wygenerowanych utworów: examples/music/

### constants
Folder zawiera pliki skryptów pythonowych, w których są zapisane stałe dla odpowiednich sieci.

### datasets
Folder zawierający pliki skryptów pythonowych, w których zaimplementowane są klasy reprezentujące dane uczące w programie wraz z potrzebnymi narzędziami do ich obróbki.

### movies_analysis.py
Skrypt służący do analizowania filmów.
Argumenty wywołania:
--dir - ścieżka od fimu, który ma być analizowany
--name - nazwa plików formacie mp4, który ma być analizowany (bez rozszerzenia)
Po wykonaniu skrypt wyświetli klasą rozpoznanych emocji.
Neutral - emocje neutralne
LALV - emocje smutne (Low Arousal Low Valency)
LAHV - emocje spokojne (Low Arousal High Valency)
HALV - emocje straszne (High Arousal Low Valency)
HAHV - emocje wesołe (High Arousal High Valency)

przykład wywołania:
python3 movies_analysis.py --dir "examples/films/" --name "ACCEDE00000"


### music_generation.py
Skrypt służący do generowania muzyki na podstawie podanych emocji i o podanej długości.
Argumenty wywołania:
--emotion - nazwa klasy emocji, na której podstawie będzie generowana muzyka
--len - długość trwania utworu (w sekundach)
--out - nazwa pliku wynikowego (plik wynikowy znajdzie się w ścieżce music_generation/out/samples/)

przykład wywołania:
python3 music_generation.py --emotion "Neutral" --len 10 --out "output"


### generate_soundtrack.py
Skrypt służący go generowania ścieżki dźwiękowej do zadanego filmu.
Argumenty wywołania:
--dir - ścieżka od fimu, który ma być analizowany
--name - nazwa plików formacie mp4, który ma być analizowany (bez rozszerzenia)
--out - nazwa pliku wynikowego (plik wynikowy znajdzie się w ścieżce music_generation/out/samples/)

przykład wywołania:
python3 movies_analysis.py --dir "examples/films/" --name "ACCEDE00000" --out "output"

### Dane uczące
Niestety z racji rozmiarów użytych zbiorów danych uczących nie jestem w stanie umieścić ich na płycie.
Dane do nauki sieci neuronowej analizującej filmy: Discrete LIRIS-ACCEDE (https://liris-accede.ec-lyon.fr/).
Dane do nauki sieci generującej muzykę: Multi-modal MIREX-like emotion dataset (http://mir.dei.uc.pt/downloads.html).

### Instrukcja konfiguracji środowiska uruchomieniowego
Aby wykonać jeden z dostępnych skryptów należy:
1. Zainstalować narzędzie pip (zarządca pakietów języka python) i kompilator język Python.
2. 

