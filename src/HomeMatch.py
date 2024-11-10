import os
import re
from typing import Dict, Optional, List
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
import chromadb
from pydantic import BaseModel, Field
from src.Configurator import Configurator

class ListingFineTuned(BaseModel):
    """
    Pydantic model for fine-tuned property listings.
    
    Attributes:
        title (str): Title of the property listing adjusted for the user's preferences.
        description (str): Description of the property listing adjusted for the user's preferences.
    """
    title: str = Field(..., description="Title of the property listing adjusted for the user's preferences.")
    description: str = Field(..., description="Description of the property listing adjusted for the user's preferences.")

class HomeMatch:
    """
    A class to manage and display an interactive form for collecting buyer preferences 
    and matching them with real estate properties.

    The class handles the entire workflow from displaying the preference collection form
    to showing matched properties based on user inputs. It uses embedding-based similarity
    search and LLM-based content adjustment to provide personalized property recommendations.

    Attributes:
        config (Configurator): Configuration object containing application settings.
        n_properties_to_find (int): Number of matching properties to find and display.
        questions (Dict[int, str]): Dictionary mapping question numbers to their text.
        responses (Dict): Dictionary storing form widget references.
        collected_responses (Dict[int, str]): Dictionary storing user's responses.
        matching_properties (List[Dict]): List of properties matched based on user preferences.
        adjusted_listings (List[ListingFineTuned]): List of property listings adjusted for user preferences.
        user_preference_text (str): Concatenated string of user preferences used for matching.
    """
    
    def __init__(self, config: Configurator, n_properties_to_find: int = 3, collected_responses: Optional[Dict[int, str]] = None):
        """
        Initialize the HomeMatch instance.

        Args:
            config (Configurator): Configuration object containing application settings.
            n_properties_to_find (int, optional): Number of matching properties to find. Defaults to 3.
            collected_responses (Optional[Dict[int, str]], optional): Pre-collected user responses. Defaults to None.
        """
        self.config = config
        self.n_properties_to_find = n_properties_to_find
        self.questions = self._load_questions()
        self.responses = {}
        self.collected_responses = collected_responses or {}

    def _load_questions(self) -> Dict[int, str]:
        """
        Load questions from text files in the configured questions folder.
        
        Returns:
            Dict[int, str]: Dictionary mapping question numbers to their text content.
        """
        question_folder = Path(self.config.get('question_folder'))
        questions = {}
        for file_path in question_folder.glob('*.txt'):
            if match := re.search(r'\d+', file_path.stem):
                q_number = int(match.group())
                questions[q_number] = file_path.read_text(encoding='utf-8').strip()
        return questions

    def _display_title(self) -> None:
        """Display the form title using configured styling."""
        title_style = self.config.get('home_match_style')['title']
        title_html = f"""
            <h2 style='
                text-align: {title_style['align']};
                color: {title_style['color']};
                font-size: {title_style['font_size']};
            '>
                {title_style['text']}
            </h2>
        """
        display(HTML(title_html))

    def _create_form(self) -> None:
        """Create and display the interactive form for collecting user preferences."""
        form_items = []
        form_style = self.config.get('home_match_style')['form']
        
        for q_number, question_text in sorted(self.questions.items()):
            question_label = widgets.HTML(
                f"<b style='font-size: {form_style['question_size']};'>{question_text}</b>",
                layout=widgets.Layout(margin=f"{form_style['margin']}")
            )
            
            response_text = self.collected_responses.get(q_number, "")
            response_box = widgets.Text(
                value=response_text,
                placeholder=form_style['placeholder'],
                layout=widgets.Layout(width=form_style['width'], padding=form_style['padding']),
                style={'description_width': 'initial'}
            )
            
            form_items.append(widgets.VBox([question_label, response_box]))
            self.responses[q_number] = response_box
        
        form_container = widgets.VBox(form_items, layout=widgets.Layout(margin=form_style['container_margin']))
        
        submit_bottom_style = self.config.get('home_match_style')['submit_button']
        submit_button = widgets.Button(
            description=submit_bottom_style['description'],
            button_style=submit_bottom_style['button_style'],
            layout=widgets.Layout(width=submit_bottom_style['width'], margin=submit_bottom_style['margin'])
        )
        submit_button.on_click(self._on_submit)
        
        display(form_container, submit_button)

    def _on_submit(self, button: widgets.Button) -> None:
        """
        Handle form submission event.
        
        Args:
            button (widgets.Button): The submit button widget that triggered the event.
        """
        responses = {q_number: widget.value.strip() for q_number, widget in self.responses.items()}
        self.collected_responses = responses
        
        clear_output()
        self._display_title()
        self.find_and_display_matching_properties()

    def find_matching_property(self):
        """
        Find matching properties using embedding-based similarity search.
        
        Updates self.matching_properties with the found matches, sorted by relevance score.
        """
        user_embedding = self._generate_user_preference_embedding()
        client = chromadb.PersistentClient(path=self.config.get("chromadb_persist_directory"))
        collection = client.get_collection(name=self.config.get('property_listings_collection'))
        
        results = collection.query(query_embeddings=[user_embedding], n_results=self.n_properties_to_find)
        matching_properties = self._parse_query_result(results)
        
        self.matching_properties = sorted(matching_properties, key=lambda x: x['score'], reverse=True)

    def _generate_user_preference_embedding(self):
        """
        Generate embedding vector for user preferences using OpenAI's embedding model.
        
        Returns:
            List[float]: Embedding vector representing user preferences.
        """
        self._prepare_user_preference_text()
        client = OpenAI(
                base_url = os.environ["OPENAI_API_BASE"],
                api_key = os.environ["OPENAI_API_KEY"]
        )
        response = client.embeddings.create(input=self.user_preference_text, model=self.config.get('embedding_model'))
        return response.data[0].embedding

    def _prepare_user_preference_text(self) -> str:
        """
        Prepare a text representation of user preferences by combining questions and responses.
        
        Updates self.user_preference_text with the formatted preference string.
        """
        self.user_preference_text = " ".join(
            f"{self.questions[q_id]}: {self.collected_responses.get(q_id, 'Not answered')}" 
            for q_id in sorted(self.questions.keys())
        )
        
    def _parse_query_result(self, results) -> List[Dict]:
        """
        Parse ChromaDB query results into a structured format.
        
        Args:
            results: Raw query results from ChromaDB.
            
        Returns:
            List[Dict]: List of dictionaries containing parsed property information.
        """
        return [
            {"id": res_id, "metadata": metadata, "score": distance, "document": doc}
            for res_id, metadata, distance, doc in zip(
                results['ids'][0], results['metadatas'][0], results['distances'][0], results['documents'][0]
            )
        ]
        
    def _load_listing_fine_tuning_prompt_template(self) -> str:
        """
        Load the prompt template for fine-tuning property listings.
        
        Returns:
            str: Content of the prompt template file.
        """
        with open(self.config.get("fine_tuning_listing_template"), "r") as f:
            return f.read()
        
    def _define_langchain_prompt_template(self, prompt_template_txt: str) -> PromptTemplate:
        """
        Create a LangChain PromptTemplate for property listings.

        Args:
            prompt_template_txt (str): Raw template text.

        Returns:
            PromptTemplate: Configured template ready for use.
        """
        return PromptTemplate(
            input_variables=self.config.get("fine_tuning_listing_input_prompt_variables"),
            template=prompt_template_txt,
        )

    def _adjust_listing_to_user_preferences(self):
        """
        Adjust property listings based on user preferences using LLM.
        
        Updates self.adjusted_listings with personalized versions of the matched properties.
        """
        parser = PydanticOutputParser(pydantic_object=ListingFineTuned)
        
        prompt_template_txt = self._load_listing_fine_tuning_prompt_template()
        fine_tuning_prompt = self._define_langchain_prompt_template(prompt_template_txt)
        
        llm = ChatOpenAI(**self.config.get("open_ai_parameters"))

        self.adjusted_listings = []
        for item in self.matching_properties: 
            query = fine_tuning_prompt.format(
                user_preference_text=self.user_preference_text,
                initial_listing=item['document'],
                format_instructions=parser.get_format_instructions()
            )
            llm_result = llm.invoke(query)
            
            result = parser.parse(llm_result.content)
            self.adjusted_listings.append(result)

    def display_adjusted_listings(self):
        """
        Display the adjusted property listings with configured styling.
        
        Shows an error message if no listings are available.
        """
        initial_message_style = self.config.get('home_match_style')['results']['initial_message']
        error_style = self.config.get('home_match_style')['results']['error']
        box_style = self.config.get('home_match_style')['results']['box']
        
        if self.adjusted_listings:
            initial_message = f"""
            <div style='text-align: {initial_message_style['text-align']}; font-size: {initial_message_style['font-size']}; 
            color: {initial_message_style['color']}; padding: {initial_message_style['padding']}; 
            background-color: {initial_message_style['background-color']}; border-radius: {initial_message_style['border-radius']}; 
            margin-bottom: {initial_message_style['margin-bottom']};'>
                {initial_message_style['text']}
            </div>
            """
            display(HTML(initial_message))
        else:
            display(HTML(f"<b style='color: {error_style['color']};'>{error_style['text']}</b>"))
            return
        
        property_widgets = []
        for listing in self.adjusted_listings:
            property_html = f"""
            <div style='background-color: {box_style['background-color']}; color: {box_style['text-color']};
            padding: {box_style['padding']}; margin: {box_style['margin']}; border: {box_style['border-size']} {box_style['border-color']};
            border-radius: {box_style['border-radius']}; font-weight: {box_style['font-weight']};'>
                <h3>{listing.title}</h3>
                <p>{listing.description}</p>
            </div>
            """
            property_widgets.append(widgets.HTML(property_html))
        
        properties_container = widgets.VBox(property_widgets)
        display(properties_container)

    def find_and_display_matching_properties(self):
        """
        Execute the complete property matching and display workflow.
        
        This method coordinates the process of finding matching properties,
        adjusting them to user preferences, and displaying the results.
        """
        self.find_matching_property()
        self._adjust_listing_to_user_preferences()
        self.display_adjusted_listings()
        
    def main(self):
        """
        Initialize and display the main interface.
        
        This method should be called to start the property matching workflow.
        """
        self._display_title()
        self._create_form()