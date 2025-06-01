# Skincare Product Recommendation System for EverGlow Labs

## 1. Overview

This project implements an AI-powered personal shopper and product recommendation system for **EverGlow Labs**, a skincare brand that emphasizes a blend of natural ingredients and scientific validation. The system is designed to understand customer queries, ask clarifying questions when necessary, and recommend suitable products from the EverGlow Labs catalog, aligning with the brand's philosophy.

The core of the system is built using Python, leveraging data processing libraries like Pandas, machine learning libraries like Scikit-learn and Sentence Transformers for embeddings and clustering, FAISS for efficient similarity search, NLTK for sentiment analysis, and LangGraph with the Gemini API for building a conversational agent with intelligent decision-making.

## 2. Brand Philosophy (EverGlow Labs)

The recommendation logic is deeply intertwined with the brand's core values:

"EverGlow Labs exists to prove that nature and science can co-author skincare that actually works. We formulate every product around three uncompromising pillars:

1.  **Plant-Powered, Clinically Proven:** 100% vegan, cruelty-free, and silicone-free, using high-potency botanicals (rosewater, cactus enzymes, oat lipids) and gold-standard actives (retinaldehyde, peptides, ceramides) validated in peer-reviewed studies. Every finished formula undergoes third-party in-vitro and in-vivo testing for efficacy and safety.
2.  **Radical Transparency:** Full-dose percentages of hero ingredients on every carton. Carbon-neutral supply chain; FSC-certified packaging and soy inks. Real, verified customer reviews only—no paid placements, no bots.
3.  **Barrier-First, Planet-First:** pH-optimized, microbiome-friendly formulas that respect your skin barrier. Cold-processed where possible to reduce energy use and preserve phyto-nutrients. 1% of revenue funds reef-safe sunscreen education and re-wilding projects.

The result: skincare, body care, and haircare that deliver visible results without compromising ethics, the environment, or your wallet."

## 3. Key Features & Functionality

* **Intelligent Query Handling:**
    * **Keyword-based Search:** Identifies specific product categories (e.g., "serums," "cleanser") mentioned by the user.
    * **Vague Query Clarification:** Asks 1-2 relevant, contextual questions to understand user needs better when queries are ambiguous (e.g., "something gentle for summer").
    * **Informational Queries:** Provides information based on product data and reviews if the user is asking a question rather than directly seeking a recommendation.
    * **"Good" Queries:** Directly proceeds to recommendations if the query is specific enough.
* **Product Recommendation:**
    * Recommends 1-3 products from the `skincare catalog.xlsx`.
    * Prioritizes recommendations by profit margin when multiple suitable products are found.
    * Provides a short justification (max 15-20 words) for each product, linking it to brand philosophy and customer needs.
* **Conversational Agent:**
    * Utilizes LangGraph to create a stateful agent that can manage multi-turn conversations.
    * Integrates with the Gemini 1.5 Flash API for natural language understanding and generation tasks (query classification, question generation, justification).
* **Data-Driven Insights:**
    * Leverages product catalog, customer reviews, and customer support interactions.
    * Performs sentiment analysis on product reviews to enrich product understanding.
    * Uses product embeddings and FAISS for efficient similarity-based product search.
    * Clusters products based on their features to identify natural groupings.
* **Human-in-the-Loop (Conceptual):** The notebook includes logic for classifying user input type which can be used to interrupt the flow and potentially involve a human if the query is neither a clear keyword nor vague, although the direct human interruption mechanism for a Next.js framework is conceptual within this notebook.

## 4. Data Sources

The system utilizes the following data files:

* `customer_support.csv`: Contains customer messages and support responses, used to extract product mentions and understand common issues.
* `product_reviews.csv`: Contains product reviews, ratings, and reviewer information, used for sentiment analysis and enriching product profiles.
* `skincare catalog.xlsx`: The primary source of product information, including product ID, name, category, description, top ingredients, tags, price, and margin.

## 5. Notebook Structure & Workflow

The Jupyter Notebook (`shopai_assignment.ipynb`) follows a structured approach:

1.  **Setup & Installation:**
    * Installs necessary Python libraries (`google-generativeai`, `langchain`, `langgraph`, `langchain-google-genai`, `wikipedia`, `sentence-transformers`, `faiss-cpu`, `nltk`).
2.  **Data Loading:**
    * Loads the three data files (`customer_support.csv`, `product_reviews.csv`, `skincare catalog.xlsx`) into Pandas DataFrames.
3.  **Data Cleaning & Preprocessing:**
    * Handles missing values and duplicates.
    * Corrects header issues in `product_reviews.csv`.
    * Standardizes product names for consistent merging.
4.  **Data Wrangling & Merging:**
    * Merges the product catalog with product reviews.
    * Extracts product mentions from customer support data and merges this information.
5.  **Feature Engineering:**
    * Creates a combined `product_text_features` column from descriptions, ingredients, categories, and tags.
    * Generates sentence embeddings for these text features using `all-MiniLM-L6-v2`.
6.  **Data Clustering:**
    * Applies K-Means clustering to the product embeddings to group similar products.
7.  **Vector Store Creation:**
    * Builds a FAISS index from the product embeddings for efficient similarity search.
    * Saves the FAISS index to `product_embeddings.faiss`.
8.  **Sentiment Analysis:**
    * Performs sentiment analysis on product reviews using NLTK's VADER.
    * Aggregates sentiment scores per product.
9.  **Comprehensive Product Profiles:**
    * Merges all processed data (catalog, reviews, sentiment, support mentions, embeddings, clusters) into a final `comprehensive_product_profiles_df`.
    * Saves this DataFrame to `comprehensive_product_profiles.csv`.
10. **LangGraph Agent Implementation:**
    * Defines a `PersonalShopperState` for the conversational agent.
    * Implements various nodes for the agent's workflow:
        * `check_input_type`: Classifies user query.
        * `identify_categories_and_tags_json`: Extracts relevant categories and tags.
        * `ask_clarification_questions`: Generates questions for vague queries.
        * `search_for_information`: Handles informational queries.
        * `recommend_products_based_on_search`: Recommends products.
    * Defines the graph structure and conditional edges for agent decision-making.
    * Provides an `agent_main` function to interact with the agent, managing chat history.

## 6. Key Libraries Used

* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For K-Means clustering.
* **Sentence Transformers:** For generating text embeddings (`all-MiniLM-L6-v2` model).
* **FAISS (faiss-cpu):** For creating and searching the vector store.
* **NLTK (VADER):** For sentiment analysis.
* **Langchain & LangGraph:** For building the conversational agent and managing its state and flow.
* **Google Generative AI (Gemini API):** For LLM-powered tasks like query classification, question generation, and justification writing.
* **Openpyxl:** (Implicitly used by Pandas for `.xlsx` files)

## 7. Setup and Installation

1.  **Clone the repository (if applicable) or ensure you have the notebook and data files.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The notebook includes commands to install libraries. You can also use the `requirements.txt` file (if generated by the notebook using `%pip freeze > requirements.txt`) for a more streamlined installation:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install them individually as shown in the notebook's initial cells:
    ```bash
    pip install -q -U pandas numpy scikit-learn sentence-transformers faiss-cpu nltk openpyxl google-generativeai langchain langgraph langchain-google-genai
    ```
4.  **Download NLTK resources:**
    The notebook should handle downloading the `vader_lexicon` for sentiment analysis. If not, you might need to run:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```
5.  **Google API Key:**
    The notebook requires a Google API Key for the Gemini model. Ensure it's set as an environment variable or directly in the code (as shown in the notebook: `os.environ["GOOGLE_API_KEY"] = 'YOUR_API_KEY'`).
    **Note:** Replace `'YOUR_API_KEY'` with your actual key. It's best practice to use environment variables for API keys.

## 8. How to Run

1.  **Ensure all data files** (`customer_support.csv`, `product_reviews.csv`, `skincare catalog.xlsx`) are in the same directory as the notebook, or update the paths in the notebook accordingly. The provided notebook seems to expect them in the `/content/` directory if running in Colab, or relative paths like `../` if the files are one level up from where the script/notebook is eventually run. Adjust as needed.
2.  **Run the Jupyter Notebook cells sequentially.**
    * The initial cells handle library installations and data loading/preprocessing.
    * The FAISS index and comprehensive profiles are saved, so subsequent runs might load these precomputed files if available.
    * The final cells demonstrate the LangGraph agent's functionality with example queries.
3.  **Interacting with the Agent:**
    The `agent_main(user_query, current_chat_history)` function in the latter part of the notebook is the primary way to interact with the recommendation system. It takes the user's query and the existing chat history as input and returns the agent's response and updated history.

## 9. Agent Logic Details

The agent's behavior is determined by the `input_type` classification:

* **`informational`:** The agent uses the `search_for_information` node to look up relevant details from product data and reviews and provides an answer.
* **`vague`:** The agent uses the `ask_clarification_questions` node to generate 1-2 questions to better understand the user's needs. The flow then ends, awaiting user response in a subsequent interaction.
* **`keyword` or `good`:**
    1.  The agent first calls `identify_categories_and_tags_json` to extract potential product categories and tags from the query.
    2.  **If categories are identified:** It proceeds to `recommend_products_based_on_search`. This node filters products strictly by the identified categories and then by identified tags (if any). Recommendations are ranked by margin.
    3.  **If NO categories are identified (even for 'keyword' or 'good' input):** It transitions to `ask_clarification_questions` because it cannot confidently recommend without a category.
    4.  **Follow-up for 'keyword' type:** If the input was 'keyword' AND categories were identified BUT no specific tags were identified (implying a broad category search), a follow-up question is generated after recommendations to narrow down preferences.

## 10. Potential Future Enhancements

* **More Sophisticated Query Understanding:** Implement more advanced NLP techniques for entity extraction (skin concerns, ingredient preferences) beyond simple keyword/tag matching.
* **Multi-Turn Clarification:** Fully implement the loop for receiving and processing answers to clarification questions within LangGraph.
* **User Profile Management:** Store user preferences and past interactions to personalize recommendations further.
* **Hybrid Recommendation Engine:** Combine content-based filtering (using embeddings) with collaborative filtering if user-item interaction data becomes available.
* **Advanced Sentiment Analysis:** Use more nuanced sentiment models and aspect-based sentiment analysis to understand what specific features users like or dislike.
* **UI Integration:** Develop a proper frontend (e.g., using Next.js as hinted in the notebook comments) for a user-friendly chat interface.
* **Error Handling and Robustness:** Add more comprehensive error handling throughout the data processing and agent interaction pipeline.
* **A/B Testing Framework:** Implement a way to test different recommendation strategies or LLM prompts.

