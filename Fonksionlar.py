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
import uuid
import asyncio

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
        "`!nesne_bilgisi` - Ekli resimdeki nesneler hakkında bilgi verir.\n"
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


async def nesne_bilgisi_komutu(ctx):
    if ctx.message.attachments:
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                unique_filename = f"{uuid.uuid4()}_{attachment.filename}"
                image_path = os.path.join(IMAGE_DIR, unique_filename)

                await attachment.save(image_path)
                try:
                    bilgi = await analyze_image_for_objects(image_path)
                    print(bilgi)  

                    if isinstance(bilgi, list):
                        objects_info = ', '.join([obj['label'] for obj in bilgi])
                        await ctx.send(f"Nesneler: {objects_info}")
                    else:
                        await ctx.send(f"API'den beklenmedik bir yanıt alındı: {bilgi}")
                except Exception as e:
                    await ctx.send(f"Bir hata oluştu: {e}")
                finally:
                    if os.path.isfile(image_path):
                        os.remove(image_path)
            else:
                await ctx.send("Lütfen geçerli bir resim dosyası ekleyin (PNG/JPG).")
    else:
        await ctx.send("Nesne bilgisi almak için lütfen bir resim ekleyin.")

async def analyze_image_for_objects(image_path):
    try:
        return await nesne_bilgisi_api(image_path)
    except Exception as e:
        return f"Bir hata oluştu: {e}"
async def nesne_bilgisi_api(image_path, retries=3, delay=30):
    headers = {
        'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
    }

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                with open(image_path, 'rb') as image_file:
                    data = aiohttp.FormData()
                    data.add_field('file', image_file, filename=os.path.basename(image_path), content_type='image/jpeg')

                    async with session.post(HUGGINGFACE_API_URL, headers=headers, data=data) as response:
                        if response.status == 200:
                            try:
                                return await response.json()
                            except aiohttp.ContentTypeError:
                                raise Exception("API response is not valid JSON")
                        elif response.status in [503, 500]:
                            print(f"Girişim {attempt + 1}/{retries} başarısız oldu: {response.status}")
                            if attempt < retries - 1:
                                await asyncio.sleep(delay)
                            else:
                                raise Exception(f"API hatası {response.status}")
                        else:
                            response_text = await response.text()
                            raise Exception(f"API hatası {response.status}: {response_text}")
        except aiohttp.ClientError as e:
            print(f"İstek hatası: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise Exception(f"{retries} girişimden sonra istek başarısız oldu: {e}")
