import os
from openai import OpenAI, APIError, RateLimitError, Timeout
from dotenv import load_dotenv
import base64

path = "COLOQUE AQUI O CAMINHO DO SEU .ENV COM A CHAVE DE REQUISIÇÕES DA OPEN AI"

load_dotenv(f"{path}.env")

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

class OpenaiDescChart:
  def __init__(self,
               client: OpenAI,
               path_image: str,
               n_temperature: float = 0.1,
               n_top_p: float = 0.9,
               n_max_tokens: int = 180
               ) -> None:
               self.client = client
               self.path_image = path_image
               self.n_temperature = n_temperature
               self.n_top_p = n_top_p
               self.n_max_tokens = n_max_tokens

  def _image_to_base64(self) -> str:
    """
    [pt] Converte uma imagem PNG em base64.
    [en] Create a image base 64 from a png image
    """
    with open(self.path_image, "rb") as f:
      return base64.b64encode(f.read()).decode("utf-8")

  def api_desc_chart(self) -> str:
    """
    [pt] Chamada da API OpenAI para descrever um gráfico.
    [en] Call OpenAI API to describe a chart.
    """

    system_prompt = "Você é um cientista de dados experiente, com atuação em risco e modelagem de crédito."

    image_desc    = """
    A imagem a seguir contém um gráfico de avaliação de modelo de crédito.

    Tarefa:
      - Avalie a ordenação e separação de risco observável ao longo dos percentis/faixas de score
      - Descreva como a métrica exibida (taxa de maus, lift, gain ou acumulado) evolui ao longo das faixas
      - Caso haja mais de um modelo/curva, compare-os apenas com base na métrica apresentada
      - Considere superior o modelo que apresenta maior separação entre faixas de maior e menor
        risco (ex.: maior lift/gain nas faixas de maior risco; menor taxa de maus nas faixas de menor risco),
        mantendo tendência consistente ao longo dos percentis.

    Restrições:
      - Não realize testes estatísticos
      - Não estime valores numéricos não fornecidos
      - Não afirme causalidade ou impacto econômico
      - Não utilize conhecimento externo ou benchmarks
      - Baseie-se exclusivamente nas evidências visuais do gráfico e na legenda/eixos.
      - Se a evidência visual for insuficiente para comparar modelos ou identificar tendência,
        declare explicitamente que é inconclusivo.

    Formato da resposta:
    - Máximo de 3 frases
    - Linguagem técnica e objetiva
    - Referencie explicitamente padrões observados no gráfico
    """

    try:
      response = self.client.responses.create(
          model="gpt-4.1",
          input=[
              {"role": "system", "content": (system_prompt)},
              {"role": "user",
              "content": [
                      {"type": "input_text",
                      "text": (image_desc)},
                      {"type": "input_image",
                      "image_url": f"data:image/png;base64,{self._image_to_base64()}"}
                    ]
                }
          ],
          temperature=self.n_temperature,
          top_p=self.n_top_p,
          max_output_tokens=self.n_max_tokens
      )

      return response.output_text.strip()

    except RateLimitError:
        return "Error: Request limit reached. Please try again later."
    except Timeout:
        return "Error: API call timeout."
    except APIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def describe_gain_credit_chart(
    client: OpenAI,
    path_image: str,
    n_temperature: float = 0.1,
    n_top_p: float = 0.9,
    n_max_tokens: int = 150
) -> str:
    """
    [pt] A função utiliza um LLM para interpretar visualmente o gráfico e gerar
    um mini-relatório técnico.
    [en] The function uses a LLM to visually interpret the chart and produce
    a technical mini-report.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI API client.
    path_image : str
        Path to the chart image to be analyzed (PNG format or equivalent).
    n_temperature : float, optional
        Degree of randomness in the model response. Lower values produce more stable
        and objective analyses. Default is 0.1.
    n_top_p : float, optional
        Probabilistic sampling parameter to control output variability.
        Used only if supported by the model. Default is 0.9.
    n_max_tokens : int, optional
        Maximum number of tokens generated in the response. Default is 150.

    Returns
    -------
    str
    """

    engine = OpenaiDescChart(
        client=client,
        path_image=path_image,
        n_temperature=n_temperature,
        n_top_p=n_top_p,
        n_max_tokens=n_max_tokens
    )

    return engine.api_desc_chart()
