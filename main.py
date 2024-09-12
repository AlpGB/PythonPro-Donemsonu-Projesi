import discord
from discord.ext import commands
from Fonksionlar import (
    yardım_komutu,
    gorsel_olustur,
    nesneleri_algila_komutu,
    ses_yazıya_dönüştür,
    yüz_algila_komutu,
    nesne_tespiti
)

intents = discord.Intents.all()
intents.messages = True
intents.message_content = True
prefix = '!'
bot = commands.Bot(command_prefix=prefix, intents=intents)

@bot.command(name='yardım')
async def yardım(ctx):
    await yardım_komutu(ctx)

@bot.command(name='foto')
async def foto(ctx, *, description: str):
    await gorsel_olustur(ctx, description)

@bot.command(name='algıla')
async def algıla(ctx):
    await nesneleri_algila_komutu(ctx)

@bot.command(name='sesten_yazıya')
async def ses_yazıya(ctx):
    await ses_yazıya_dönüştür(ctx)

@bot.command(name='yüz_algıla')
async def yüz_algıla(ctx):
    await yüz_algila_komutu(ctx)

@bot.command(name='nesne_tespiti')
async def nesne_tespiti(ctx):
    await nesne_tespiti(ctx)

bot.run('MTI3NDMwNDI1NTM1MTg0OTA0MQ.GEWkvj.i7gZcvlGyLFff_b6EyvujnKsl3qU-xH7wua2_U')
