from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEndpoint
from tavily import TavilyClient
from quickstart_connect import connect_to_database
from astrapy import Database, Collection
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set API keys
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
tavily_client = TavilyClient(api_key="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")

# Instantiate the LLM
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-3B-Instruct", task="text-generation")

# Connect to Astra DB
database = connect_to_database()

# Set the collection name
collection_name = "emailllmdata"

# Helper function to create or retrieve a collection
def get_or_create_collection(database: Database, collection_name: str) -> Collection:
    try:
        collection = database.get_collection(collection_name)
        logging.info(f"Collection {collection_name} retrieved successfully.")
    except Exception:
        collection = database.create_collection(collection_name)
        logging.info(f"Collection {collection_name} created successfully.")
    return collection

# Retrieve or create the collection
collection = get_or_create_collection(database, collection_name)

# Utility function to truncate or summarize content
def truncate_or_summarize(content: str, max_length: int = 512) -> str:
    """Truncate or summarize content to fit within the maximum token limit."""
    if len(content) > max_length:
        logging.warning(f"Content length {len(content)} exceeds max_length {max_length}. Truncating.")
        return content[:max_length]  # Simple truncation
    return content

@app.route('/generate-report', methods=['POST'])
def generate_report():
    data = request.json
    company_url = data.get('company_url')
    # report_query = data.get('report_query')

    if not company_url:
        return jsonify({'error': 'Missing company_url or report_query'}), 400

    try:
        # Fetch content using Tavily
        try:
            content = tavily_client.search(company_url, search_depth="advanced", topic="news", time_range="day")["results"]
        except Exception as e:
            logging.error(f"Error fetching content: {str(e)}")
            return jsonify({'error': 'Failed to fetch content'}), 500

        # Truncate or summarize content
        truncated_content = truncate_or_summarize(content, max_length=512)

        # Construct the prompt
        prompt = (
            "Provide a comprehensive and in-depth analysis of the {company_url}. Ensure the information is up-to-date "
            "and organized in a structured, reader-friendly format. Use bullet points, tables, and visuals (if applicable) "
            "to enhance clarity and readability.\n\n"
            "Include the following sections with detailed insights:\n\n"
            "### 1. Company Overview\n"
            "- **Founding and History**:\n"
            "  - Year of establishment and founder(s).\n"
            "  - Significant milestones, acquisitions, or expansions.\n"
            "- **Mission and Vision**:\n"
            "  - Mission statement, core values, and long-term vision.\n"
            "- **Industry and Market Position**:\n"
            "  - Description of the industry it operates in.\n"
            "  - Current market share or rank within the industry.\n"
            "  - Overview of key markets (regional or global).\n"
            "- **Products and Services**:\n"
            "  - List of flagship products or services.\n"
            "  - Revenue contribution by product/service line (if available).\n"
            "- **Organizational Structure**:\n"
            "  - Overview of hierarchy (e.g., divisions, regions, or subsidiaries).\n"
            "- **Subsidiaries and Affiliates**:\n"
            "  - Names and descriptions of key subsidiaries.\n"
            "  - Strategic partnerships or joint ventures.\n\n"
            "### 2. Financial Performance\n"
            "- **Revenue and Profitability**:\n"
            "  - Year-on-year revenue growth trends.\n"
            "  - Operating income, net income, and margins.\n"
            "- **Market Capitalization** (for public companies):\n"
            "  - Current market cap and recent trends.\n"
            "- **Debt and Leverage**:\n"
            "  - Debt-to-equity ratio, key borrowing levels.\n"
            "- **Key Financial Ratios**:\n"
            "  - Price-to-Earnings (P/E) ratio, Return on Equity (ROE), Return on Assets (ROA), etc.\n"
            "- **Recent Financial News**:\n"
            "  - Notable updates from quarterly or annual reports.\n"
            "  - Mergers, acquisitions, or major financial disclosures.\n\n"
            "### 3. Competitive Landscape\n"
            "- **Major Competitors**:\n"
            "  - Key companies competing in the same market.\n"
            "  - Comparative analysis of market share, products, and pricing.\n"
            "- **Competitive Advantages**:\n"
            "  - Unique selling propositions (USPs) or strengths.\n"
            "- **Competitive Challenges**:\n"
            "  - Areas where the company is at a disadvantage.\n"
            "- **SWOT Analysis**:\n"
            "  - Strengths, Weaknesses, Opportunities, Threats.\n"
            "- **Porter’s Five Forces Analysis**:\n"
            "  - Supplier power, buyer power, competitive rivalry, threat of substitutes, and threat of new entrants.\n\n"
            "### 4. Leadership and Management\n"
            "- **Key Executives**:\n"
            "  - Names, roles, and brief biographies of top executives (CEO, CFO, CTO, etc.).\n"
            "- **Leadership Style**:\n"
            "  - Description of the leadership approach and notable strategies.\n"
            "- **Board of Directors**:\n"
            "  - Key members and their backgrounds.\n\n"
            "### 5. Technology and Innovation\n"
            "- **R&D Activities**:\n"
            "  - Areas of investment and key projects in research and development.\n"
            "- **Intellectual Property**:\n"
            "  - Number of patents, trademarks, or other IP assets.\n"
            "- **Technological Advancements**:\n"
            "  - Significant innovations or breakthroughs in recent years.\n"
            "- **Adoption of AI and Emerging Technologies**:\n"
            "  - Use of AI, machine learning, or other advanced technologies.\n\n"
            "### 6. Social and Environmental Impact\n"
            "- **Corporate Social Responsibility (CSR)**:\n"
            "  - Key initiatives, philanthropic activities, or community projects.\n"
            "- **Sustainability Practices**:\n"
            "  - Efforts in reducing carbon footprint, renewable energy use, waste management.\n"
            "- **Ethical Considerations**:\n"
            "  - Compliance with regulations, controversies, or ethical dilemmas.\n\n"
            "### 7. Latest News and Developments\n"
            "- **Recent News**:\n"
            "  - Summaries of major announcements, partnerships, or controversies.\n"
            "- **Industry Trends**:\n"
            "  - Broader trends impacting the company or its industry.\n"
            "- **Upcoming Events**:\n"
            "  - Notable milestones or events in the pipeline.\n"
            f"\nInformation: \"\"\"{truncated_content}\"\"\"\n\n"
        )

        # Generate the report
        report = llm.invoke(prompt)

        # Prepare data for Astra DB
        vectorize_input = truncate_or_summarize(f"{company_url}  {report}")
        document = {
            "company_url": company_url,
            # "report_query": report_query,
            "report": report,
            "$vectorize": vectorize_input
        }

        # Insert into Astra DB
        inserted = collection.insert_one(document)
        document_id = inserted.inserted_id
        logging.info(f"Data inserted with ID: {document_id}")

        return jsonify({
            'message': 'Report generated and stored successfully',
            'report': report,
            'document_id': document_id
        })

    except Exception as e:
        logging.error(f"Failed to generate or store report: {str(e)}")
        return jsonify({'error': f"Failed to generate or store report: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
