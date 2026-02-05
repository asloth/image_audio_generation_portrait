from openai import OpenAI
from pydantic import BaseModel, Field
import json
from audio import record_audio, play_audio
import base64


validation_instructions = """
Eres un asistente especializado en criminología y técnicas forenses que ayuda a recopilar información precisa para crear un retrato hablado. 
Tu objetivo es obtener una descripción visual detallada del sujeto.

Analiza la conversación y determina qué información falta del usuario.
Información requerida:
1. Rasgos Generales y Complexión:
    Género y edad aproximada.
    Forma del rostro (ovalado, cuadrado, redondo, etc.).
    Tono de piel y complexión física (delgada, atlética, robusta).
2. Ojos y Parte Superior:
    Color, tamaño y forma de los ojos.
    Cejas (pobladas, finas, arqueadas).
    Uso de accesorios (lentes, gorras, piercings en cejas).
3. Nariz y Boca:
    Forma y tamaño de la nariz (respingada, ancha, aguilena).
    Grosor de los labios y tamaño de la boca.
    Características dentales o vello facial (bigote, barba, candado).
4. Cabello y Orejas:
    Color, textura (lacio, rizado) y longitud del cabello.
    Tipo de corte o peinado.
    Tamaño de las orejas o uso de aretes.
5. Señas Particulares:
    Cicatrices, lunares, tatuajes visibles o manchas en la piel.

Si falta información, haz una pregunta específica sobre el dato que falta.
Si el usuario indica que no recuerda o no sabe, acepta esa respuesta y continúa con la siguiente información faltante.

Si tienes toda la información, establece datos_completos en true.
IMPORTANTE EXCEPCION> Si el usuario te informa que no tiene mas informacion sobre el retrato/sujeto, establece datos_completos en true.

"""

image_instruc = """Eres un experto en generar imagenes de retratos hablados basados en descripciones detalladas.
Usa la información proporcionada para crear un retrato hablado detallado y preciso del sujeto descrito.
{json_data}
La informacion que falta tienes que inferirla o inventarla basandote en la informacion proporcionada.
Usa el estilo indicado para crear una imagen coherente y realista.
"""

client = OpenAI()

class Characteristics(BaseModel):
    genero: str | None = Field(description="Género del sujeto")
    edad_aproximada: str | None= Field(description="Edad aproximada del sujeto")
    forma_rostro: str | None= Field(description="Forma del rostro")
    tono_piel: str | None= Field(description="Tono de piel")
    complexion_fisica: str | None= Field(description="Complexión física")
    color_ojos: str | None= Field(description="Color de los ojos")
    tamano_ojos: str | None= Field(description="Tamaño de los ojos")
    forma_ojos: str | None= Field(description="Forma de los ojos")
    cejas: str | None= Field(description="Descripción de las cejas")
    accesorios_ojos: str | None= Field(description="Accesorios usados en los ojos")
    forma_nariz: str | None= Field(description="Forma de la nariz")
    tamano_nariz: str | None= Field(description="Tamaño de la nariz")
    grosor_labios: str | None= Field(description="Grosor de los labios")
    tamano_boca: str | None= Field(description="Tamaño de la boca")
    caracteristicas_dentales: str | None= Field(description="Características dentales o vello facial")
    color_cabello: str | None= Field(description="Color del cabello")
    textura_cabello: str | None= Field(description="Textura del cabello")
    longitud_cabello: str | None= Field(description="Longitud del cabello")
    tipo_corte_peinado: str | None= Field(description="Tipo de corte o peinado")
    tamano_orejas: str | None= Field(description="Tamaño de las orejas")
    uso_aretes: str | None= Field(description="Uso de aretes")
    senas_particulares: str | None= Field(description="Señas particulares como cicatrices, lunares, tatuajes")

class ValidateData(BaseModel):
    datos_completos: bool = Field(description="Indica si se ha recopilado toda la información necesaria o si el usuario no tiene más datos para proporcionar")
    charact_data: Characteristics = Field(description="Datos recopilados para el retrato hablado")
    estilo_preferido: str = Field(description="Estilo realista o caricatura para el retrato hablado")
    pregunta_siguiente: str = Field(description="Siguiente pregunta para recopilar información faltante")

messages = []

# Enhanced cost tracking structure
cost_usage = {
    'total_tokens': 0,
    'total_cost': 0.0,
    'details': [],
    'summary': {
        'audio_transcription': {'tokens': 0, 'cost': 0.0, 'count': 0},
        'text_generation': {'tokens': 0, 'cost': 0.0, 'count': 0},
        'audio_synthesis': {'chars': 0, 'cost': 0.0, 'count': 0},
        'image_generation': {'images': 0, 'cost': 0.0, 'count': 0},
        'moderation': {'tokens': 0, 'cost': 0.0, 'count': 0}
    }
}

# Pricing per 1M tokens or per unit
cost_per_1M_tokens = {
    "gpt-5": 1.25,
    "gpt-5-nano": 0.05,
    "omni-moderation-latest": 0.0,
    "gpt-4o-transcribe_audio_input": 6.0,
    "gpt-4o-mini-tts_per_1M_chars": 15.0,  # $15 per 1M characters
    "gpt-image-1_1024x1024_high": 0.167  # $0.167 per image for 1024x1024 high quality
}

text_model = "gpt-5-nano"
trans_model = "gpt-4o-transcribe"
image_model = "gpt-image-1"
audio_model = "gpt-4o-mini-tts"

def track_usage(model_name, usage_type, **kwargs):
    """
    Centralized function to track usage and costs
    
    Args:
        model_name: Name of the model used
        usage_type: Type of usage ('transcription', 'text', 'tts', 'image', 'moderation')
        **kwargs: Additional parameters like tokens, chars, images count
    """
    if usage_type == 'transcription':
        tokens = kwargs.get('tokens', 0)
        cost = (tokens / 1_000_000) * cost_per_1M_tokens[f"{model_name}_audio_input"]
        
        cost_usage['total_tokens'] += tokens
        cost_usage['total_cost'] += cost
        cost_usage['summary']['audio_transcription']['tokens'] += tokens
        cost_usage['summary']['audio_transcription']['cost'] += cost
        cost_usage['summary']['audio_transcription']['count'] += 1
        
        cost_usage['details'].append({
            'model': model_name,
            'type': 'audio_transcription',
            'tokens': tokens,
            'cost': cost
        })
        
    elif usage_type == 'text':
        tokens = kwargs.get('tokens', 0)
        cost = (tokens / 1_000_000) * cost_per_1M_tokens[model_name]
        
        cost_usage['total_tokens'] += tokens
        cost_usage['total_cost'] += cost
        cost_usage['summary']['text_generation']['tokens'] += tokens
        cost_usage['summary']['text_generation']['cost'] += cost
        cost_usage['summary']['text_generation']['count'] += 1
        
        cost_usage['details'].append({
            'model': model_name,
            'type': 'text_generation',
            'tokens': tokens,
            'cost': cost
        })
        
    elif usage_type == 'tts':
        chars = kwargs.get('chars', 0)
        cost = (chars / 1_000_000) * cost_per_1M_tokens[f"{model_name}_per_1M_chars"]
        
        cost_usage['total_cost'] += cost
        cost_usage['summary']['audio_synthesis']['chars'] += chars
        cost_usage['summary']['audio_synthesis']['cost'] += cost
        cost_usage['summary']['audio_synthesis']['count'] += 1
        
        cost_usage['details'].append({
            'model': model_name,
            'type': 'audio_synthesis',
            'chars': chars,
            'cost': cost
        })
        
    elif usage_type == 'image':
        size = kwargs.get('size', '1024x1024')
        quality = kwargs.get('quality', 'high')
        n_images = kwargs.get('n', 1)
        
        # Cost per image based on size and quality
        cost_per_image = cost_per_1M_tokens[f"{model_name}_{size}_{quality}"]
        cost = cost_per_image * n_images
        
        cost_usage['total_cost'] += cost
        cost_usage['summary']['image_generation']['images'] += n_images
        cost_usage['summary']['image_generation']['cost'] += cost
        cost_usage['summary']['image_generation']['count'] += 1
        
        cost_usage['details'].append({
            'model': model_name,
            'type': 'image_generation',
            'size': size,
            'quality': quality,
            'n_images': n_images,
            'cost_per_image': cost_per_image,
            'total_cost': cost
        })
        
    elif usage_type == 'moderation':
        tokens = kwargs.get('tokens', 0)
        cost = (tokens / 1_000_000) * cost_per_1M_tokens[model_name]
        
        cost_usage['total_tokens'] += tokens
        cost_usage['total_cost'] += cost
        cost_usage['summary']['moderation']['tokens'] += tokens
        cost_usage['summary']['moderation']['cost'] += cost
        cost_usage['summary']['moderation']['count'] += 1
        
        cost_usage['details'].append({
            'model': model_name,
            'type': 'moderation',
            'tokens': tokens,
            'cost': cost
        })

while True:
    record_audio("record.wav")
    audio = open("record.wav", "rb")
    audio_response = client.audio.transcriptions.create(
        model=trans_model,
        file=audio,
        response_format="json"
    )
    audio.close()
    
    # Track transcription usage
    track_usage(trans_model, 'transcription', tokens=audio_response.usage.total_tokens)
    
    # Print transcribed text
    print(f"User: {audio_response.text}")
    
    input_text = audio_response.text
    data_moderation = client.moderations.create(input=input_text, model="omni-moderation-latest")
    
    # Track moderation usage (if it has usage info)
    if hasattr(data_moderation, 'usage') and data_moderation.usage:
        track_usage("omni-moderation-latest", 'moderation', tokens=data_moderation.usage.total_tokens)

    if data_moderation.results[0].flagged:
        print("Lo siento, tu mensaje ha sido marcado por contenido inapropiado. Por favor, intenta de nuevo.")
        continue
        
    # Usar primer modelo para validar datos con JSON schema
    messages.append({"content": input_text, "role": "user"})
    validation_response = client.responses.parse(
        model=text_model,
        input=messages,
        instructions=validation_instructions,
        text_format=ValidateData,
        reasoning={"effort": "minimal"}
    )
    
    # Track text generation usage
    track_usage(text_model, 'text', tokens=validation_response.usage.total_tokens)
    
    # Obtener datos validados y guardarlos
    validation_data = validation_response.output_parsed
    messages.append({"content": validation_response.output_text, "role": "assistant"})
    print("Datos recopilados hasta ahora:", validation_data.charact_data.model_dump())

    if validation_data and validation_data.datos_completos:
        # Generar respuesta de audio indicando que se generará el retrato hablado
        response = client.audio.speech.create(
            model=audio_model,
            input="Ahora generaré el retrato hablado basado en la información proporcionada.",
            speed=1.3,
            voice="onyx",
            instructions="Habla en español",
            response_format="wav"
        )
        
        # Track TTS usage
        chars = len("Ahora generaré el retrato hablado basado en la información proporcionada.")
        track_usage(audio_model, 'tts', chars=chars)
        
        # Save and play audio response
        with open("response.wav", "wb") as f:
            f.write(response.content)
        play_audio("response.wav")
        # Guardar datos en un archivo JSON
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(validation_data.model_dump(), f, ensure_ascii=False, indent=4)

        # Usar segundo modelo para crear el retrato hablado
        image_response = client.images.generate(
            model=image_model,
            prompt=image_instruc.format(json_data=json.dumps(validation_data.charact_data.model_dump())),
            n=1,
            quality="high",
            size="1024x1024",
        )

        # Track image generation usage
        track_usage(image_model, 'image', size="1024x1024", quality="high", n=1)

        if image_response:
            image = image_response.data[0].b64_json
            b64_image = base64.b64decode(image)
        
            try:
                # Save usage tracking with summary
                with open("tokens.json", "w", encoding="utf-8") as jf:
                    json.dump(cost_usage, jf, ensure_ascii=False, indent=4)
                    
                with open("portrait.png", "wb") as f:
                    f.write(b64_image)
                print("El retrato hablado ha sido generado y guardado como 'portrait.png'.")
            except Exception as e:
                print(f"❌ Error al guardar el archivo: {e}")
        # Respuesta final de audio
        response = client.audio.speech.create(
            model=audio_model,
            input="Gracias por toda la información proporcionada. He generado el retrato hablado basado en los datos recopilados.",
            speed=1.3,
            voice="onyx",
            instructions="Habla en español",
            response_format="wav"
        )
        
        # Track TTS usage
        chars = len("Gracias por toda la información proporcionada. He generado el retrato hablado basado en los datos recopilados.")
        track_usage(audio_model, 'tts', chars=chars)
        
        # Save and play audio response
        with open("response.wav", "wb") as f:
            f.write(response.content)
        play_audio("response.wav")
        break
    else:
        # Generar respuesta de audio para la siguiente pregunta
        response = client.audio.speech.create(
            model=audio_model,
            input=validation_data.pregunta_siguiente,
            speed=1.3,
            voice="onyx",
            instructions="Habla en español",
            response_format="wav"
        )
        
        # Track TTS usage
        chars = len(validation_data.pregunta_siguiente)
        track_usage(audio_model, 'tts', chars=chars)
        
        # Save and play audio response
        with open("response.wav", "wb") as f:
            f.write(response.content)
        play_audio("response.wav")
