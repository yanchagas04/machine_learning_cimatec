# src/engine.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DEEPFACE_LOG_LEVEL'] = '40'

import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("absl").setLevel(logging.FATAL)

from deepface import DeepFace
import cv2

class FaceRecognizer:
    """
    Classe responsável por verificar se uma pessoa é autorizada.
    """

    def __init__(self, authorized_path: str):
        if not os.path.exists(authorized_path):
            raise FileNotFoundError(f"O caminho da pasta de autorizados não foi encontrado: {authorized_path}")
        self.authorized_path = authorized_path
        self.authorized_faces = []
    
    def CheckFace(self, img_path: str) -> bool:
        """
        Verifica se a pessoa na imagem é autorizada.
        """
        try:
            dfs = DeepFace.find(
                img_path=img_path,
                db_path=self.authorized_path,
                enforce_detection=True,
                model_name="Facenet",
                detector_backend="opencv",
                silent=True
                )
            if len(dfs) > 0 and not dfs[0].empty:
                return True
            return False
        except ValueError as ve:
            print(f"Aviso interno no CheckFace: {ve}")
            return False
        except Exception as e:
            print(f"Erro no CheckFace: {e}")
            return False


    def ShowResult(self, face_obj, img, color, text):
        facial_area = face_obj['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def VerificationLoop(self) -> None:
        while True:
            img_path = input("Digite o caminho da imagem que deseja testar (ou 'q' para sair): ")
            if img_path.lower() == 'q':
                break
            img_path = img_path.strip('"\'')
            if not os.path.isfile(img_path):
                print(f"Erro: O arquivo '{img_path}' não foi encontrado.")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print("Erro: Não foi possível carregar a imagem. Verifique o caminho.")
                continue
            filename = os.path.basename(img_path)
            try:
                # Usando MTCNN que é muito mais poderoso que o OpenCV padrão
                face_objs = DeepFace.extract_faces(img_path=img_path, enforce_detection=True, detector_backend="mtcnn")
                is_authorized = self.CheckFace(img_path)
                if is_authorized:
                    color = (0, 255, 0) # Verde
                    text = "ACESSO LIBERADO"
                else:
                    color = (0, 0, 255) # Vermelho
                    text = "ACESSO NEGADO"
                for face_obj in face_objs:
                    self.ShowResult(face_obj, img, color, text)

                cv2.imshow(f"Analise: {filename}", img)
                print(f"Imagem '{filename}' analisada. Pressione qualquer tecla na janela da imagem para continuar...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except ValueError as ve:
                print(f"Nenhum rosto encontrado na imagem {filename}. Erro: {ve}")
            except Exception as e:
                print(f"Erro inesperado com a imagem {filename}: {e}")
                