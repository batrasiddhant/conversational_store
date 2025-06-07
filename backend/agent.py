import os
from typing import List, Dict, Any, TypedDict, Tuple # Added Tuple
import json # Ensure json is imported
import re # Ensure re is imported

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer
from typing import Union

# Load a pre-trained model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDlo-rq9YxnUs4r5sJEHlgMw1dgs5TAd08' # Replace with your key or use environment variables

# --- Load data (ensure paths are correct) ---
try:
    index = faiss.read_index('../product_embeddings.faiss')
    comprehensive_product_profiles_df = pd.read_csv('../comprehensive_product_profiles.csv')

    tag_set = set()
    if 'tags' in comprehensive_product_profiles_df.columns and comprehensive_product_profiles_df.tags.notna().any():
        for tags_entry in comprehensive_product_profiles_df.tags.dropna().str.split('|'):
            tag_set.update(tags_entry)
    else:
        print("Warning: 'tags' column missing or empty in comprehensive_product_profiles.csv. Tags functionality will be limited.")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}. Please ensure '../product_embeddings.faiss' and '../comprehensive_product_profiles.csv' exist.")
    # Depending on the desired behavior, you might want to exit or run with limited functionality.
    # For now, we'll let it potentially fail later if these are crucial and not found.
    index = None
    comprehensive_product_profiles_df = pd.DataFrame() # Empty DataFrame
    tag_set = set()


# --- State Definition with Chat History ---
class PersonalShopperState(TypedDict):
    input: str  # User's original query
    chat_history: Union[List[Dict[str, str]], None]  # e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    input_type: Union[str, None]  # 'keyword', 'vague', 'good', or 'informational'
    clarification_questions: Union[List[str], None]  # Questions to ask the user
    clarification_answers: Union[Dict[str, str], None]  # User's answers to questions (used if user provides answers later)
    recommended_products: Union[List[Dict[str, Any]], None]  # List of recommended products
    follow_up_question: Union[str, None]  # Question to ask after recommendations for keyword inputs
    informational_answer: Union[str, None]
    identified_categories: Union[List[str], None]
    identified_tags: Union[List[str], None]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

# --- Helper function to format chat history ---
def format_chat_history_for_prompt(chat_history: Union[List[Dict[str, str]], None]) -> str:
    if not chat_history or len(chat_history) <= 1:  # Don't show history if it's just the current user query
        return "No significant prior conversation history."
    
    # Display all but the last message (which is the current user input already in 'input_text')
    history_to_display = chat_history[:-1] if chat_history else []
    if not history_to_display:
        return "No significant prior conversation history."

    history_str = "Conversation History:\n"
    for entry in history_to_display:
        history_str += f"{entry['role'].capitalize()}: {entry['content']}\n"
    return history_str.strip()

# --- Node Functions (Updated for Chat History) ---

def identify_categories_and_tags_json(state: PersonalShopperState) -> PersonalShopperState:
    print(f"Identifying categories and tags for input: {state['input']}")
    input_text = state['input']
    chat_history_str = format_chat_history_for_prompt(state.get('chat_history'))

    available_categories = []
    if 'category' in comprehensive_product_profiles_df.columns:
        available_categories = comprehensive_product_profiles_df.category.unique().tolist()
    available_tags = list(tag_set)

    prompt = f"""{chat_history_str}

You are a helpful assistant that identifies potential skincare product categories and tags from the LATEST user's query, considering the conversation history for context.
Based on the LATEST query, list the most likely relevant categories and tags *ONLY* from the following provided lists.

Available Categories: {', '.join(available_categories) if available_categories else "N/A"}
Available Tags: {', '.join(available_tags) if available_tags else "N/A"}

Respond with a JSON object containing two keys: "categories" and "tags". Each key should have a list of strings.
If no relevant categories or tags from the provided lists are identified for a key, the list should be empty.

Example JSON output if the user asks for "hydrating serum for oily skin":
{{
  "categories": ["Serum"],
  "tags": ["Hydration", "oily-skin"]
}}

LATEST User Query: "{input_text}"

JSON Output:
"""
    new_state = state.copy()
    new_state.update({
        'recommended_products': None, 'follow_up_question': None,
        'informational_answer': None, 'clarification_questions': None,
        'identified_categories': None, 'identified_tags': None
    })

    try:
        response = llm.invoke(prompt)
        json_output_str = response.content.strip()
        # Robust JSON extraction
        try:
            # Try to find the JSON block if markdown is used
            match = re.search(r"```json\s*([\s\S]*?)\s*```", json_output_str)
            if match:
                json_data_str = match.group(1)
            else: # Assume raw JSON or cleanup if LLM doesn't use markdown
                json_data_str = json_output_str[json_output_str.find('{'):json_output_str.rfind('}')+1]

            identified_terms_json = json.loads(json_data_str)
            categories = identified_terms_json.get('categories', [])
            tags = identified_terms_json.get('tags', [])

            filtered_categories = [cat.strip() for cat in categories if cat.strip() in available_categories]
            filtered_tags = [tag.strip() for tag in tags if tag.strip() in available_tags]

            new_state['identified_categories'] = filtered_categories if filtered_categories else None
            new_state['identified_tags'] = filtered_tags if filtered_tags else None

            print(f"Identified categories: {new_state['identified_categories']}")
            print(f"Identified tags: {new_state['identified_tags']}")

        except json.JSONDecodeError:
            print(f"LLM returned invalid JSON: {json_output_str}.")
        except Exception as e_parse:
             print(f"Error parsing LLM JSON output: {e_parse}. Output: {json_output_str}")


    except Exception as e:
        print(f"An error occurred during category/tag identification: {e}.")

    return new_state


def check_input_type(state: PersonalShopperState) -> PersonalShopperState:
    print(f"Checking input type for: {state['input']} using LLM")
    input_text = state['input']
    chat_history_str = format_chat_history_for_prompt(state.get('chat_history'))

    # MODIFIED PROMPT STARTS HERE
    prompt = f"""

Conversation History: "{chat_history_str}"

You are an expert assistant at classifying user queries in an ongoing conversation. Your task is to classify the intent of LATEST user skincare query (provided at the end) considering the context provided above into one of four types: 'keyword', 'vague', 'good', or 'informational'.

**Crucially, you MUST analyze the ENTIRE "Conversation History" provided above.** This history is vital for understanding the user's current intent, especially if the LATEST query is a response to a question from the assistant or a refinement of a previously discussed topic.

Instructions for using conversation history:
1.  **Identify Active Context:** Determine if there's an active product category (e.g., "serum," "moisturizer") or specific topic being discussed from the recent history.
2.  **Check for Answers/Refinements:** If the LATEST User Query seems to be an answer to a question asked by the assistant in the previous turn, or provides a specific detail (like a skin concern or preferred attribute) related to the active context, combine this information.
    * Example 1:
        * History: Assistant asked, "What are your main concerns for the serum?"
        * LATEST User Query: "anti-aging and brightness"
        * Interpretation: The user wants an "anti-aging and brightening serum." This should be classified as 'good'.
    * Example 2:
        * History: User said, "I need a moisturizer." Assistant recommended some.
        * LATEST User Query: "preferably one for sensitive skin"
        * Interpretation: The user wants a "moisturizer for sensitive skin." This is 'good'.
3.  **Standalone Queries:** If the LATEST User Query introduces a completely new topic or doesn't directly relate to the immediate preceding turns, classify it more on its own merits, but still be aware of the overall conversation flow.

Definitions of Classification Types (apply after considering history):
- 'keyword': The LATEST User Query, possibly contextualized by history, primarily names a specific product category (e.g., "serum", "cleanser"). Example: "Let's look at cleansers now." or (History: "We discussed serums.") LATEST User Query: "Okay, just a basic one." (interpreted as basic serum -> keyword/good).
- 'vague': The LATEST User Query, even when considering the conversation history, remains too general, lacks a clear product type, or doesn't provide enough specifics to make a recommendation. Example: User: "I don't know what I want." or User: "Something else." (without further context).
- 'good': The LATEST User Query, when combined with necessary context from history, is specific enough to recommend products. It typically implies a product type and one or more attributes/concerns. Example: (History: "Looking for sunscreen.") LATEST User Query: "SPF 50 and good for oily skin." (interpreted as "SPF 50 sunscreen for oily skin").
- 'informational': The LATEST User Query seeks information, advice, or comparison, rather than directly requesting a product type for recommendation. Example: "What's the difference between retinol and bakuchiol?" or "Are those eye creams any good?".

LATEST User Query: "{input_text}"

Based on your analysis of the conversation history and the LATEST User Query, output ONLY one word corresponding to its classification: 'keyword', 'vague', 'good', or 'informational'.

Classification:"""



    # MODIFIED PROMPT ENDS HERE

    input_type = 'vague' # Changed default to 'vague' as it's often safer if LLM fails.
    try:
        response = llm1.invoke(prompt)
        classified_type = response.content.strip().lower()
        # Ensure the response is *exactly* one of the allowed words.
        if classified_type in ['keyword', 'vague', 'good', 'informational']:
            input_type = classified_type
            print(f"LLM classified input as: {input_type}")
        else:
            print(f"LLM returned unexpected output for input type classification: '{classified_type}'. Input was: '{input_text}'. Using fallback rules.")
            # Fallback logic remains the same, but now triggered if LLM output is not strictly one of the keywords.
            if "how" in input_text.lower() or "what" in input_text.lower() or "?" in input_text or "explain" in input_text.lower() or "tell me about" in input_text.lower():
                input_type = 'informational'
            elif any(kw in input_text.lower() for kw in ['serum', 'moisturizer', 'cleanser', 'toner', 'spf', 'mask', 'cream', 'oil', 'lotion', 'sunscreen', 'exfoliator', 'balm']):
                if len(input_text.split()) > 2 or any(concern_kw in input_text.lower() for concern_kw in ['oily', 'dry', 'sensitive', 'anti-aging', 'acne', 'brightening', 'hydrating', 'aging', 'wrinkles', 'pores', 'redness']):
                    input_type = 'good'
                else:
                    input_type = 'keyword'
            elif len(input_text.split()) <= 3 and ('something' in input_text.lower() or 'anything' in input_text.lower()):
                 input_type = 'vague'
            else: # Default fallback
                input_type = 'vague'
            print(f"Fallback classification used: {input_type}")

    except Exception as e:
        print(f"An error occurred during LLM classification call: {e}. Defaulting to 'vague'.")
        input_type = 'vague'

    new_state = state.copy()
    new_state['input_type'] = input_type
    return new_state


def ask_clarification_questions(state: PersonalShopperState) -> PersonalShopperState:
    print("Generating clarification questions for vague input...")
    input_text = state['input']
    chat_history_str = format_chat_history_for_prompt(state.get('chat_history'))

    prompt = f"""{chat_history_str}

You are a helpful and friendly salesperson for EverGlow Labs.
The customer's LATEST query is: "{input_text}". This seems a bit vague, or we need more details based on our conversation.
To help them find the perfect product, generate 1-2 concise clarification questions.
Focus on understanding their skin type, specific concerns, or preferred product types (like serum, toner, SPF) if not already clear from the history.
Present them clearly.

Clarification Questions:
"""
    questions = []
    try:
        response = llm.invoke(prompt)
        questions = [q.strip() for q in response.content.strip().split('\n') if q.strip() and q.strip() != "Clarification Questions:"]
        print(f"Generated questions: {questions}")
    except Exception as e:
        print(f"Error generating clarification questions: {e}")
        questions = ["Could you tell me a bit more about your skin concerns or what you're looking for?"]


    new_state = state.copy()
    new_state['clarification_questions'] = questions
    new_state['recommended_products'] = None
    new_state['follow_up_question'] = None
    new_state['informational_answer'] = None
    return new_state

# search_products_with_index remains the same as it doesn't directly use LLM or chat history for its core FAISS logic.
def search_products_with_index(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index with the query and returns product details.
    Assumes access to comprehensive_product_profiles_df, index, and embedding_model globals.
    """
    print(f"Searching FAISS index for query: '{query}'")
    # Ensure globals are accessible or passed if this were in a class/different structure
    global comprehensive_product_profiles_df, index, embedding_model

    if comprehensive_product_profiles_df is None or comprehensive_product_profiles_df.empty or index is None or embedding_model is None:
        print("Error: FAISS index, embedding model, or product data not available for search.")
        return []

    try:
        query_embedding = embedding_model.encode(query).astype('float32')
        distances, indices_faiss = index.search(query_embedding.reshape(1, -1), top_k)
        result_indices = indices_faiss[0]

        valid_indices = [idx for idx in result_indices if idx < len(comprehensive_product_profiles_df)]
        if not valid_indices:
             print("No valid indices found in FAISS search.")
             return []

        search_results_df = comprehensive_product_profiles_df.iloc[valid_indices].copy()
        search_results = search_results_df.to_dict('records')

        print(f"Found {len(search_results)} initial potential products from search.")
        return search_results

    except Exception as e:
        print(f"An error occurred during FAISS search: {e}")
        return []


def search_for_information(state: PersonalShopperState) -> PersonalShopperState:
    print(f"Searching for informational answer for query: {state['input']}")
    query = state['input']
    chat_history_str = format_chat_history_for_prompt(state.get('chat_history'))

    global comprehensive_product_profiles_df

    if comprehensive_product_profiles_df is None or comprehensive_product_profiles_df.empty :
        print("Error: Comprehensive product profiles data not available for informational search.")
        new_state = state.copy()
        new_state['informational_answer'] = "Sorry, I can't access product information right now."
        return new_state

    query_keywords = query.lower().split()
    relevant_snippets = []
    max_snippets = 3

    # Simple keyword search (can be improved with embedding-based semantic search on text fields too)
    for idx, row in comprehensive_product_profiles_df.iterrows():
        product_info = f"{row.get('name','')} {row.get('description','')} {row.get('top_ingredients','')} {row.get('category','')}".lower()
        if any(keyword in product_info for keyword in query_keywords):
            relevant_snippets.append(f"- From product '{row.get('name','Unknown Product')}': {row.get('description','No description')[:150]}...")
            if len(relevant_snippets) >= max_snippets:
                break
    
    if len(relevant_snippets) < max_snippets and 'Review' in comprehensive_product_profiles_df.columns:
        # Filter reviews that mention any keyword
        try:
            # Ensure query_keywords are not empty and handle regex special characters
            safe_keywords = [re.escape(kw) for kw in query_keywords if kw]
            if safe_keywords:
                relevant_reviews_df = comprehensive_product_profiles_df[
                    comprehensive_product_profiles_df['Review'].str.lower().str.contains('|'.join(safe_keywords), na=False)
                ]
                for _, row in relevant_reviews_df.head(max_snippets - len(relevant_snippets)).iterrows():
                    review_text = row['Review']
                    # Attempt to find a snippet around a keyword
                    found_snippet = review_text[:150] # Fallback to start of review
                    for kw in query_keywords:
                        if kw in review_text.lower():
                            match = re.search(r'(?i).{0,70}' + re.escape(kw) + r'.{0,70}', review_text)
                            if match:
                                found_snippet = match.group(0)
                                break
                    relevant_snippets.append(f"- From review for '{row.get('name','Unknown Product')}': \"...{found_snippet}...\"")
        except Exception as e_review_search:
            print(f"Error during review search: {e_review_search}")


    informational_answer = "I couldn't find specific information related to your query in our product data or reviews."
    if relevant_snippets:
        snippets_text = "\n".join(relevant_snippets)
        answer_prompt = f"""{chat_history_str}

You are a helpful and friendly salesperson for EverGlow Labs.
The customer's LATEST query is: "{query}"
Based on this query and the conversation history, use the following relevant information snippets from our product data and customer reviews to provide a concise and helpful answer.
Cite the source of the information if appropriate (e.g., "From product...", "A review mentions...").

Relevant Information:
{snippets_text}

Answer:"""
        try:
            response = llm.invoke(answer_prompt)
            informational_answer = response.content.strip()
            print(f"Generated informational answer:\n{informational_answer}")
        except Exception as e:
            print(f"Error generating informational answer with LLM: {e}")
            informational_answer = "Sorry, I found some information but couldn't formulate a perfect answer right now."
    else:
        print("No relevant snippets found for informational query.")

    new_state = state.copy()
    new_state['informational_answer'] = informational_answer
    new_state.update({'recommended_products': None, 'clarification_questions': None, 'follow_up_question': None})
    return new_state


def recommend_products_based_on_search(state: PersonalShopperState) -> PersonalShopperState:
    print("Attempting to recommend products based on identified categories and tags.")
    identified_categories = state.get('identified_categories', [])
    identified_tags = state.get('identified_tags', [])
    chat_history_str = format_chat_history_for_prompt(state.get('chat_history'))
    user_input = state['input']


    global comprehensive_product_profiles_df
    new_state = state.copy()
    new_state.update({'clarification_questions': None, 'informational_answer': None, 'follow_up_question': None})


    if comprehensive_product_profiles_df is None or comprehensive_product_profiles_df.empty:
        print("Error: Product data not available for recommendation.")
        new_state['recommended_products'] = [{"name": "Sorry", "justification": "Product data is currently unavailable."}]
        return new_state

    filtered_df = comprehensive_product_profiles_df.copy()

    if not identified_categories and not identified_tags:
        print("No categories or tags identified. Cannot recommend based on strict filtering. Consider if this path should lead to clarification.")
        # This path might indicate a logic flaw if 'good' or 'keyword' input led here without identified terms.
        # For now, providing a generic message. Or, one could try a broader embedding search.
        # Let's try a general FAISS search based on the input if no categories/tags
        print(f"Performing a general FAISS search for: {user_input}")
        search_results_direct = search_products_with_index(user_input, top_k=3)
        if search_results_direct:
             filtered_df = pd.DataFrame(search_results_direct)
             print(f"Found {len(filtered_df)} products via direct FAISS search as fallback.")
        else:
            new_state['recommended_products'] = [{"name": "Sorry", "justification": "I couldn't find specific products based on your request. Could you try rephrasing or adding more details?"}]
            return new_state
    else:
        if identified_categories:
            print(f"Strictly filtering by categories: {identified_categories}")
            filtered_df = filtered_df[filtered_df['category'].isin(identified_categories)]
        else: # No categories identified, but maybe tags were. If neither, previous block handles.
            if not identified_tags: # Should not happen if the above "if not identified_categories and not identified_tags" is hit
                print("No categories identified by LLM step. Cannot strictly filter by category.")
                # This implies the graph logic from identify_categories_and_tags should have gone to 'ask_clarification'
                # However, if it reaches here, it means a 'good' or 'keyword' input didn't yield categories.
                # A fallback: If no categories but input was 'good', do a general search.
                if state.get('input_type') in ['good', 'keyword']: # Fallback for good/keyword inputs if no category found
                    print(f"Fallback: Performing general FAISS search for '{user_input}' as no categories were strictly identified but input type suggested recommendation.")
                    search_results_fallback = search_products_with_index(user_input, top_k=3)
                    if search_results_fallback:
                        filtered_df = pd.DataFrame(search_results_fallback)
                    else:
                        new_state['recommended_products'] = [{"name": "Sorry", "justification": "Could not identify a relevant product category or find matches for your request."}]
                        return new_state
                else: # Should ideally be handled by graph routing to clarification
                    new_state['recommended_products'] = [{"name": "Sorry", "justification": "Could not identify a relevant product category from your request."}]
                    return new_state
        
        if identified_tags and not filtered_df.empty:
            print(f"Filtering by tags: {identified_tags}")
            try:
                # Handle potential NaN in 'tags' column and ensure it's string
                valid_tags_series = filtered_df['tags'].fillna('').astype(str).str.lower()
                # Create a boolean mask
                mask = valid_tags_series.apply(lambda x_tags: any(tag.lower() in x_tags for tag in identified_tags))
                filtered_df = filtered_df[mask]
            except Exception as e_tag_filter:
                print(f"Error during tag filtering: {e_tag_filter}")


    if filtered_df.empty:
        justification_text = "Could not find products matching your specific criteria."
        if identified_categories and identified_tags:
             justification_text = f"Could not find products matching categories ({', '.join(identified_categories)}) and tags ({', '.join(identified_tags)})."
        elif identified_categories:
             justification_text = f"Could not find products for category ({', '.join(identified_categories)}) with the specified tags."
        elif identified_tags:
             justification_text = f"Could not find products with tags ({', '.join(identified_tags)}) in the initially considered categories."

        new_state['recommended_products'] = [{"name": "Sorry", "justification": justification_text}]
        return new_state

    # --- Ranking by Margin (if available) and selecting top 3 ---
    if 'margin' in filtered_df.columns:
        # Ensure margin is numeric, coercing errors to NaN, then fill NaN with a low value (e.g., 0 or -1) for sorting
        filtered_df['margin'] = pd.to_numeric(filtered_df['margin'], errors='coerce').fillna(0)
        recommended_products_df = filtered_df.sort_values(by='margin', ascending=False).head(3)
    else:
        recommended_products_df = filtered_df.head(3)

    recommended_products_output = []
    if not recommended_products_df.empty:
        for _, row in recommended_products_df.iterrows():
            justification_prompt = f"""{chat_history_str}

You are a helpful and friendly salesperson for EverGlow Labs.
The customer's LATEST query was: "{user_input}"
We are recommending the product: "{row.get('name', 'N/A')}"
Product Details:
- Category: {row.get('category', 'N/A')}
- Description: {row.get('description', 'N/A')[:150]}...
- Key Ingredients: {row.get('top_ingredients', 'N/A')}
- Tags: {row.get('tags', 'N/A')}

Based on the conversation and product details, provide a *very short* (10-20 words) justification for this recommendation. Highlight a key benefit.
Justification:
"""
            justification = "This product is a great choice from EverGlow Labs, aligning with your needs." # Fallback
            try:
                response = llm.invoke(justification_prompt)
                justification = response.content.strip()
                # Ensure justification is short
                justification = ' '.join(justification.split()[:20]) + ('...' if len(justification.split()) > 20 else '')
            except Exception as e:
                print(f"Error generating justification for {row.get('name', 'N/A')}: {e}")

            recommended_products_output.append({
                "name": row.get('name', 'N/A'),
                "justification": justification,
                "category": row.get('category', 'N/A'),
                "price": row.get('price', 'N/A'), # Ensure 'price' column exists and is formatted
                "description_snippet": row.get('description', '')[:100] + "..."
            })
    else: # Should be covered by earlier empty check, but as a safeguard
        new_state['recommended_products'] = [{"name": "Sorry", "justification": "I found some potential products but couldn't finalize recommendations."}]
        return new_state

    new_state['recommended_products'] = recommended_products_output

    # --- Follow-up question logic (simplified based on whether tags were fully utilized) ---
    # If categories were ID'd, but either no tags were ID'd OR ID'd tags didn't narrow down much from category.
    if new_state.get('identified_categories') and not new_state.get('identified_tags') and recommended_products_output:
        cat_list_str = ", ".join(new_state['identified_categories'])
        follow_up_q_prompt = f"""{chat_history_str}

You are a helpful and friendly salesperson for EverGlow Labs.
The customer was interested in products from the category: {cat_list_str}. You've just recommended some.
To further refine or confirm their choice, ask a brief, open-ended follow-up question that invites them to share more about their specific needs or preferences for these {cat_list_str} (e.g., specific skin concerns like hydration, anti-aging, texture preferences, etc., if not already clear).
Keep it concise (1 sentence).

Follow-up Question:
"""
        try:
            response = llm.invoke(follow_up_q_prompt)
            new_state['follow_up_question'] = response.content.strip()
        except Exception as e:
            print(f"Error generating follow-up question: {e}")
            new_state['follow_up_question'] = f"Is there anything specific you're looking for in a {cat_list_str.lower()} (like texture or a particular benefit)?"

    return new_state


# --- Graph Definition ---
workflow = StateGraph(PersonalShopperState)

workflow.add_node("check_input", check_input_type)
workflow.add_node("identify_categories_and_tags", identify_categories_and_tags_json)
workflow.add_node("ask_clarification", ask_clarification_questions)
workflow.add_node("search_information", search_for_information)
workflow.add_node("recommend_products", recommend_products_based_on_search)

workflow.set_entry_point("check_input")

workflow.add_conditional_edges(
    "check_input",
    lambda state: state['input_type'],
    {
        "informational": "search_information",
        "vague": "ask_clarification", # If vague, always ask clarification first
        "keyword": "identify_categories_and_tags", # For keyword, try to identify first
        "good": "identify_categories_and_tags",    # For good, try to identify first
    }
)

workflow.add_conditional_edges(
    "identify_categories_and_tags",
    # If categories AND/OR tags are identified, try to recommend.
    # If NEITHER are identified (even after this step for 'keyword'/'good' inputs), then ask for clarification.
    lambda state: "recommend_products" if (state.get('identified_categories') or state.get('identified_tags')) else "ask_clarification",
    {
        "recommend_products": "recommend_products",
        "ask_clarification": "ask_clarification",
    }
)

workflow.add_edge("ask_clarification", END)
workflow.add_edge("search_information", END)
workflow.add_edge("recommend_products", END) # Follow-up is handled within the node if needed

app = workflow.compile()

# --- Main Agent Function (Updated for Chat History) ---
def agent_main(user_query: str, current_chat_history: Union[List[Dict[str, str]], None] = None):
    if current_chat_history is None:
        current_chat_history = []

    # The agent's internal state will use this combined history for the current turn
    turn_chat_history = current_chat_history + [{"role": "user", "content": user_query}]

    initial_state_params = {
        "input": user_query,
        "chat_history": turn_chat_history, # Pass the history including the latest user query
        "input_type": None,
        "clarification_questions": None,
        "clarification_answers": None, # This would be populated if user answers questions in a multi-turn vague flow
        "recommended_products": None,
        "follow_up_question": None,
        "informational_answer": None,
        "identified_categories": None,
        "identified_tags": None
    }

    # Invoke the graph
    final_state = app.invoke(initial_state_params)

    # Construct a single text response from the agent's actions
    assistant_response_parts = []
    if final_state.get('informational_answer'):
        assistant_response_parts.append(final_state['informational_answer'])
    
    if final_state.get('clarification_questions'):
        # assistant_response_parts.append("I have a couple of questions to help you better:")
        for q_idx, q_text in enumerate(final_state['clarification_questions']):
            assistant_response_parts.append(f"{q_text}") # Present questions directly
            
    if final_state.get('recommended_products'):
        if not (len(final_state['recommended_products']) == 1 and final_state['recommended_products'][0].get('name', '').lower() == "sorry"):
            assistant_response_parts.append("\nHere are some products I recommend:")
            for prod in final_state['recommended_products']:
                recommendation = f"- **{prod.get('name', 'N/A')}**"
                if 'justification' in prod and prod['justification']:
                    recommendation += f": {prod['justification']}"
                
                details = []
                if 'category' in prod and prod.get('category'):
                    details.append(f"Category: {prod['category']}")
                if 'price' in prod and prod.get('price') is not None:
                    try:
                        details.append(f"Price: ${float(prod['price']):.2f}")
                    except (ValueError, TypeError):
                        details.append(f"Price: {prod['price']}") # Keep as is if not convertible
                if details:
                    recommendation += f" ({', '.join(details)})"
                assistant_response_parts.append(recommendation)
        else: # It's a "Sorry" message
            assistant_response_parts.append(final_state['recommended_products'][0].get('justification', "Sorry, I couldn't find a suitable product."))

    if final_state.get('follow_up_question'):
        # Add a newline if there were recommendations before the follow-up
        if final_state.get('recommended_products') and not (len(final_state['recommended_products']) == 1 and final_state['recommended_products'][0].get('name', '').lower() == "sorry"):
            assistant_response_parts.append(f"\n{final_state['follow_up_question']}")
        else:
            assistant_response_parts.append(final_state['follow_up_question'])


    if not assistant_response_parts:
        # This case should ideally be rare if graph logic is sound and nodes always produce some output or error.
        # Check input type to provide a more relevant default response.
        input_type = final_state.get('input_type', 'unknown')
        if input_type == 'vague' and not final_state.get('clarification_questions'):
             assistant_response_parts.append("I understand. To help you best, could you please tell me a bit more about your specific skin concerns or what kind of product you're looking for?")
        elif input_type == 'informational' and not final_state.get('informational_answer'):
             assistant_response_parts.append("I'm sorry, I couldn't find the specific information you're looking for right now.")
        elif final_state.get('recommended_products') is None and input_type in ['good', 'keyword']: # No recs, no sorry message
            assistant_response_parts.append("I'm having a bit of trouble finding products for that right now. Could you try a different search?")
        else:
             assistant_response_parts.append("I'm not quite sure how to help with that. Could you please rephrase or provide more details?")


    assistant_response_str = "\n".join(filter(None,assistant_response_parts)).strip()


    # Append assistant's response to history for the next turn
    # This is the history that should be passed to the next call of agent_main
    updated_full_chat_history = turn_chat_history
    # + [{"role": "assistant", "content": assistant_response_str}]
    print(updated_full_chat_history)

    return {
        'assistant_response': assistant_response_str,
        'chat_history': updated_full_chat_history,
        'clarification_questions': final_state.get('clarification_questions'),
        'recommended_products': final_state.get('recommended_products'),
        'follow_up_question': final_state.get('follow_up_question'),
        'informational_answer': final_state.get('informational_answer'),
        'input_type': final_state.get('input_type')
    }

if __name__ == '__main__':
    # --- Example Usage of the Chat Agent ---
    print("Personal Shopper Agent Initialized. Type 'exit' to end.")
    chat_history = []