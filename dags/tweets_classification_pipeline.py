import json
import pendulum
from airflow.decorators import dag, task
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


class Sentiment(BaseModel):
    sentiment: str = Field(description="tweet sentiment")
        
@dag(
    dag_id="tweets_classification_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def tweets_classification_pipeline():

    @task()
    def extract():

        """
        #### Extract task

        """
        splits = {
            'train': 'sentiment/train-00000-of-00001.parquet',
            'test': 'sentiment/test-00000-of-00001.parquet',
            'validation': 'sentiment/validation-00000-of-00001.parquet'
        }

        df = pd.read_parquet(
            "hf://datasets/cardiffnlp/tweet_eval/" + splits["train"],
            engine="pyarrow",
            columns=None,  # optional: limit to specific columns
            filters=None,  # optional: row filtering (not for head)
        )

        df = df.head(5)
        return df[['text']]
    
    @task(multiple_outputs=True)
    def transform(tweets_df: pd.DataFrame):
        """
        #### Transform task
        Classifies the sentiment of tweets using Ollama.
        """
        # Define the sentiment classification prompt
        template = """Classify the sentiment of the following tweet as Positive, Negative, or Neutral:

        Tweet: {tweet}

        Sentiment:"""
        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOllama(model="smollm")
        structured_llm = llm.with_structured_output(Sentiment)
        # Classify sentiment for each tweet
        sentiments = []
        for tweet in tweets_df['text']:
            chain = prompt | structured_llm
            response = chain.invoke({"tweet": tweet})
            sentiments.append(response.sentiment)

        # Add sentiment classification to the DataFrame
        tweets_df['sentiment'] = sentiments
        return {"classified_tweets": tweets_df.to_dict(orient="records")}

    @task()
    def load(classified_tweets: list):
        """
        #### Load task
        Prints only the first classified tweet with its sentiment and saves all to a JSON file.
        """
        # Define the output file path
        output_file = "/workspaces/depi-eyouth/output/classified_tweets.json"

        # Save the classified tweets to a JSON file
        with open(output_file, "w") as f:
            json.dump(classified_tweets, f, indent=4)

        print(f"Classified tweets saved to {output_file}")

        # Print only the first tweet with its sentiment
        if classified_tweets:
            first_tweet = classified_tweets[0]
            print(f"First Tweet: {first_tweet['text']}, Sentiment: {first_tweet['sentiment']}")

    tweets_df = extract()
    classified_tweets = transform(tweets_df)
    load(classified_tweets["classified_tweets"])

tweets_classification_pipeline()
