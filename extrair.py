import zipfile
import os

for i in range(14,24):
    # Caminho para o arquivo ZIP no Google Drive
    zip_file_path = '/home/marcelo/Desktop/dataset/test/test-2/test80_'+str(i)+'.zip'  # Substitua com o caminho correto

    password = '.chalearnLAPFirstImpressionsSECONDRoundICPRWorkshop2016.'

    # Pasta de destino para a extração
    extract_folder = '/home/marcelo/Desktop/dataset/test/test-2/test80_' + str(i)

    # Criar a pasta de destino (se não existir)

    os.makedirs(extract_folder, exist_ok=True)

    # Extrair o conteúdo do arquivo ZIP
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder, pwd=password.encode('utf-8'))

    # Listar os arquivos extraídos
    extracted_files = os.listdir(extract_folder)
    print(f'Arquivos extraídos para a pasta "{extract_folder}":')
    for file in extracted_files:
        print(file)
