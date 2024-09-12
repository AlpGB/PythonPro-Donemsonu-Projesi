import discord
import os
import requests
import time
import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import aiohttp

IMAGE_DIR = "/Users/evrimbayrak/Desktop/AI Bot Folder/"
HUGGINGFACE_API_KEY = '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'
HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2'

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()
transform = T.Compose([T.ToTensor()])

def nesne_algila_ve_ciz(image_path, output_image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0)
    with torch.no_grad():
        tahminler = model(image_tensor)
    for i, kutu in enumerate(tahminler[0]['boxes']):
        skor = tahminler[0]['scores'][i].item()
        if skor > 0.5:
            kutu = kutu.tolist()
            cv2.rectangle(image, (int(kutu[0]), int(kutu[1])), (int(kutu[2]), int(kutu[3])), color=(0, 255, 0), thickness=3)
    cv2.imwrite(output_image_path, image)

def api_den_gorsel_olustur(description, retries=3, delay=30):
    headers = {
        'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {'inputs': description}
    for attempt in range(retries):
        try:
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)
            if response.status_code == 200:
                return response.content
            elif response.status_code in [503, 500]:
                error_message = response.json().get('error', 'Bilinmeyen hata')
                print(f"Girişim {attempt + 1}/{retries} başarısız oldu: {error_message}")
                if attempt < retries - 1:
                    print(f"{delay} saniye içinde tekrar deneniyor...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Hata {response.status_code}: {error_message}")
            elif response.status_code == 400:
                error_message = response.json().get('error', 'Bilinmeyen hata')
                print(f"Müşteri hatası 400: {error_message}")
                raise Exception(f"Hata 400: {error_message}")
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"İstek hatası: {e}")
            if attempt < retries - 1:
                print(f"{delay} saniye içinde tekrar deneniyor...")
                time.sleep(delay)
            else:
                raise Exception(f"{retries} girişimden sonra istek başarısız oldu: {e}")

async def yardım_komutu(ctx):
    yardım_mesajı = (
        "Komutlar:\n"
        "`!foto <açıklama>` - Stable Diffusion kullanarak açıklamaya dayalı bir görüntü oluşturur.\n"
        "`!algıla` - Ekli resimdeki nesneleri algılar.\n"
        "`!sesten_yazıya` - Ekli ses dosyasını metne dönüştürür.\n"
        "`!nesne_tespiti` - Ekli resimdeki nesneler hakkında bilgi verir.\n"
        "`!yüz_algıla` - Ekli resimdeki yüzleri algılar."
    )
    await ctx.send(yardım_mesajı)

async def gorsel_olustur(ctx, description: str):
    try:
        output_image_path = os.path.join(IMAGE_DIR, "oluşturulan_görsel.png")
        image_data = api_den_gorsel_olustur(description)
        with open(output_image_path, 'wb') as image_file:
            image_file.write(image_data)
        await ctx.send("İşte oluşturduğunuz görsel:", file=discord.File(output_image_path))
        if os.path.isfile(output_image_path):
            os.remove(output_image_path)
    except Exception as e:
        await ctx.send(f"Görsel oluştururken bir hata oluştu: {e}")

async def nesneleri_algila_komutu(ctx):
    if ctx.message.attachments:
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(IMAGE_DIR, attachment.filename)
                output_image_path = os.path.join(IMAGE_DIR, f"algılanan_{attachment.filename}")
                await attachment.save(image_path)
                try:
                    nesne_algila_ve_ciz(image_path, output_image_path)
                    await ctx.send("Nesneler algılandı!", file=discord.File(output_image_path))
                except Exception as e:
                    await ctx.send(f"Bir hata oluştu: {e}")
                finally:
                    if os.path.isfile(image_path):
                        os.remove(image_path)
                    if os.path.isfile(output_image_path):
                        os.remove(output_image_path)
            else:
                await ctx.send("Lütfen geçerli bir resim dosyası ekleyin (PNG/JPG).")
    else:
        await ctx.send("Nesne algılama için lütfen bir resim ekleyin.")

async def ses_yazıya_dönüştür(ctx):
    if ctx.message.attachments:
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(('mp3', 'wav', 'm4a')):
                audio_path = os.path.join(IMAGE_DIR, attachment.filename)
                await attachment.save(audio_path)
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
                        'Content-Type': 'application/json'
                    }
                    with open(audio_path, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    data = {'audio': audio_data}
                    async with session.post('https://api-inference.huggingface.co/models/openai/whisper-small', headers=headers, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            transcription = result['text']
                            await ctx.send(f"Yazıya döküm: {transcription}")
                        else:
                            await ctx.send(f"Ses dosyasını yazıya dökerken bir hata oluştu: {response.status}")
                if os.path.isfile(audio_path):
                    os.remove(audio_path)
            else:
                await ctx.send("Lütfen geçerli bir ses dosyası ekleyin (MP3/WAV/M4A).")
    else:
        await ctx.send("Yazıya döküm için lütfen bir ses dosyası ekleyin.")

async def yüz_algila_komutu(ctx):
    if ctx.message.attachments:
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(IMAGE_DIR, attachment.filename)
                output_image_path = os.path.join(IMAGE_DIR, f"yüz_algılanan_{attachment.filename}")
                await attachment.save(image_path)
                try:
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.imwrite(output_image_path, image)
                    await ctx.send("Yüzler algılandı!", file=discord.File(output_image_path))
                except Exception as e:
                    await ctx.send(f"Bir hata oluştu: {e}")
                finally:
                    if os.path.isfile(image_path):
                        os.remove(image_path)
                    if os.path.isfile(output_image_path):
                        os.remove(output_image_path)
            else:
                await ctx.send("Lütfen geçerli bir resim dosyası ekleyin (PNG/JPG).")
    else:
        await ctx.send("Yüz algılama için lütfen bir resim ekleyin.")


async def nesne_tespiti(ctx):
    if ctx.message.attachments:
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(IMAGE_DIR, attachment.filename)
                await attachment.save(image_path)
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        predictions = model(image_tensor)
                    labels = predictions[0]['labels']
                    scores = predictions[0]['scores']
                    object_names = {
                        1: 'kişi', 2: 'bisiklet', 3: 'araba', 4: 'motor', 5: 'uçak', 6: 'otobüs',
                        7: 'tren', 8: 'kamyon', 9: 'bot', 10: 'trafik lambası', 11: 'yangın musluğu',
                        12: 'zebra geçidi', 13: 'park sayacı', 14: 'bank', 15: 'kuş', 16: 'kedi',
                        17: 'köpek', 18: 'at', 19: 'koyun', 20: 'inek', 21: 'fil', 22: 'ayı', 23: 'yılan',
                        24: 'sandalye', 25: 'koltuk', 26: 'çanta', 27: 'şemsiye', 28: 'top', 29: 'frizbi',
                        30: 'kaykay', 31: 'surf tahtası', 32: 'tenis raketi', 33: 'şişe', 34: 'şarap kadehi',
                        35: 'fincan', 36: 'çatal', 37: 'bıçak', 38: 'kaşık', 39: 'kase', 40: 'muz',
                        41: 'klavye', 42: 'sandviç', 43: 'portakal', 44: 'brokoli', 45: 'havuç', 46: 'hotdog',
                        47: 'pizza', 48: 'donut', 49: 'kek', 50: 'klavye', 51: 'telefon', 52: 'uzaktan kumanda',
                        53: 'elma', 54: 'fare', 55: 'dizüstü bilgisayar', 56: 'monitör', 57: 'ekmek',
                        58: 'tenis topu', 59: 'kamera', 60: 'mikrodalga', 61: 'fırın', 62: 'buzdolabı',
                        63: 'tost makinesi', 64: 'mutfak lavabosu', 65: 'yatak', 66: 'masa', 67: 'tuvalet',
                        68: 'bilgisayar', 69: 'çeşme', 70: 'gitar', 71: 'motosiklet kaskı', 72: 'beyzbol sopası',
                        73: 'şapka', 74: 'kitap', 75: 'mum', 76: 'yorgan', 77: 'perde', 78: 'oyuncak', 79: 'ayna',
                        80: 'duş başlığı'
                    }
                    detected_objects = []
                    for i, score in enumerate(scores):
                        if score > 0.8:
                            label = labels[i].item()
                            object_name = object_names.get(label, 'bir nesne')
                            detected_objects.append(object_name)
                    if detected_objects:
                        response = "Resimde şunları buldum: " + ', '.join(detected_objects)
                    else:
                        response = "Resimde belirgin bir nesne bulamadım."
                    await ctx.send(response)
                except Exception as e:
                    await ctx.send(f"Bir hata oluştu: {e}")
                finally:
                    if os.path.isfile(image_path):
                        os.remove(image_path)
            else:
                await ctx.send("Lütfen geçerli bir resim dosyası ekleyin (PNG/JPG).")
    else:
        await ctx.send("Nesne bilgisi için lütfen bir resim ekleyin.")
