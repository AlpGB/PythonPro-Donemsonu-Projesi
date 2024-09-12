import discord
from discord.ext import commands
import os
import requests
import time
import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
from googletrans import Translator
import io
import aiohttp

intents = discord.Intents.all()
intents.messages = True
intents.message_content = True
prefix = '!'
bot = commands.Bot(command_prefix=prefix, intents=intents)

IMAGE_DIR = "/Users/evrimbayrak/Desktop/AI Bot Folder/"

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

transform = T.Compose([
    T.ToTensor()
])

HUGGINGFACE_API_KEY = 'HUGGING_FACE_API_KEY'
HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2'

translator = Translator()

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
    data = {
        'inputs': description,
    }
    
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

@bot.event
async def on_ready():
    print(f'{bot.user} artık çevrimiçi')

@bot.command(name='yardım')
async def yardım_komutu(ctx):
    yardım_mesajı = (
        "Ben, görüntü oluşturma ve nesne algılama konusunda yardımcı olabilen bir botum.\n"
        "Komutlar:\n"
        "`!foto <açıklama>` - Stable Diffusion kullanarak açıklamaya dayalı bir görüntü oluşturur.\n"
        "`!algıla` - Ekli resimdeki nesneleri algılar.\n"
        "`!çeviri <dil kodu> <metin>` - Metni belirtilen dile çevirir.\n"
        "`!sesten yazıya` - Ekli ses dosyasını metne dönüştürür."
    )
    await ctx.send(yardım_mesajı)

@bot.command(name='foto')
async def gorsel_olustur(ctx, *, açıklama: str):
    try:
        output_image_path = os.path.join(IMAGE_DIR, "oluşturulan_görsel.png")
        image_data = api_den_gorsel_olustur(açıklama)
        
        with open(output_image_path, 'wb') as image_file:
            image_file.write(image_data)

        await ctx.send("İşte oluşturduğunuz görsel:", file=discord.File(output_image_path))

        if os.path.isfile(output_image_path):
            os.remove(output_image_path)
    
    except Exception as e:
        await ctx.send(f"Görsel oluştururken bir hata oluştu: {e}")

@bot.command(name='algıla')
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

@bot.command(name='çeviri')
async def metni_cevir(ctx, hedef_dil: str, *, metin: str):
    try:
        çeviri = translator.translate(metin, dest=hedef_dil)
        await ctx.send(f"Çeviri: {çeviri.text}")
    except Exception as e:
        await ctx.send(f"Metni çevirirken bir hata oluştu: {e}")
        await ctx.send("Olası hataları düzeltme yolları:")
        await ctx.send("Doğru format: !çeviri <çevirmek istediğin dil> <çevirilecek cümle>")
        await ctx.send("Çevirmek istediğin dili Ingilizce ile yaz.")
        await ctx.send("Çevriliecek cümleyi yada çevirmek istediğin dili yanlış yazmadığından emin ol.")

@bot.command(name='sesten yazıya')
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

bot.run('BOT_TOKEN')
