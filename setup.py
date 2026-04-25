from setuptools import setup, find_packages

setup(
    name="earnings_qa",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "faiss-cpu",
        "numpy",
        "requests",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "earnings-qa=earnings_qa.cli.interface:main",
        ],
    },
    author="Financial Analyst",
    description="A scalable, metadata-driven Hybrid Financial RAG Chatbot.",
)
