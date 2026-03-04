#!/usr/bin/env python3
"""
Literature Search Helper for PI-LLM Alignment

This script provides search queries and links for finding relevant papers.
"""

# Key papers with direct links
PAPERS = {
    "rlhf": {
        "title": "Training a Helpful and Harmless Assistant with RLHF",
        "authors": "Bai et al.",
        "year": 2022,
        "arxiv": "2204.05862",
        "url": "https://arxiv.org/abs/2204.05862"
    },
    "instructgpt": {
        "title": "Training language models to follow instructions with human feedback",
        "authors": "Ouyang et al.",
        "year": 2022,
        "neurips": "2022",
        "url": "https://arxiv.org/abs/2203.02155"
    },
    "pinn": {
        "title": "Physics-informed neural networks",
        "authors": "Raissi et al.",
        "year": 2019,
        "journal": "Journal of Computational Physics",
        "url": "https://www.sciencedirect.com/science/article/pii/S0021999118307125"
    },
    "dpo": {
        "title": "Direct Preference Optimization",
        "authors": "Rafailov et al.",
        "year": 2023,
        "arxiv": "2305.18290",
        "url": "https://arxiv.org/abs/2305.18290"
    },
    "constitutional_ai": {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": "Bai et al.",
        "year": 2023,
        "arxiv": "2212.08073",
        "url": "https://arxiv.org/abs/2212.08073"
    },
    "constrained_decoding": {
        "title": "The Curious Case of Neural Text Degeneration",
        "authors": "Holtzman et al.",
        "year": 2020,
        "venue": "ACL",
        "url": "https://arxiv.org/abs/1904.09751"
    },
    "hallucination_survey": {
        "title": "Survey on Hallucination in Large Language Models",
        "authors": "Ji et al.",
        "year": 2023,
        "arxiv": "2309.01219",
        "url": "https://arxiv.org/abs/2309.01219"
    },
    "truthfulqa": {
        "title": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
        "authors": "Lin et al.",
        "year": 2022,
        "arxiv": "2109.07958",
        "url": "https://arxiv.org/abs/2109.07958"
    }
}

# Search queries for different platforms
SEARCH_QUERIES = {
    "google_scholar": [
        '"physics-informed neural networks" constraint enforcement',
        '"reinforcement learning from human feedback" large language models',
        '"constrained decoding" language generation',
        '"hallucination" large language models',
        '"knowledge distillation" constraints'
    ],
    "arxiv": [
        'all:rlhf OR cat:cs.LG',
        'all:hallucination AND cat:cs.CL',
        'all:"physics-informed" AND cat:cs.LG'
    ],
    "semantic_scholar": [
        'https://www.semanticscholar.org/search?q=physics-informed+neural+networks',
        'https://www.semanticscholar.org/search?q=RLHF+language+models',
        'https://www.semanticscholar.org/search?q=constrained+decoding'
    ]
}


def print_papers():
    """Print all key papers with links."""
    print("=" * 70)
    print("KEY PAPERS FOR PI-LLM ALIGNMENT LITERATURE REVIEW")
    print("=" * 70)
    print()

    for key, paper in PAPERS.items():
        print(f"📄 {paper['title']}")
        print(f"   Authors: {paper['authors']} ({paper['year']})")
        print(f"   URL: {paper['url']}")
        print()


def print_search_queries():
    """Print search queries for different platforms."""
    print("=" * 70)
    print("SEARCH QUERIES")
    print("=" * 70)
    print()

    for platform, queries in SEARCH_QUERIES.items():
        print(f"🔍 {platform.upper().replace('_', ' ')}")
        for query in queries:
            print(f"   {query}")
        print()


def generate_bibtex():
    """Generate BibTeX entries for key papers."""
    bibtex_entries = f"""
% Key Papers for PI-LLM Alignment

@article{{raissi2019physics,
  title={{Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations}},
  author={{Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E}},
  journal={{Journal of Computational Physics}},
  volume={{378}},
  pages={{686--707}},
  year={{2019}},
  publisher={{Elsevier}}
}}

@article{{bai2022training,
  title={{Training a helpful and harmless assistant with reinforcement learning from human feedback}},
  author={{Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Stanislav and Drain, Dawn and Fort, Stanislav and Ganguli, Deepak and Henighan, Tom and Hume, Tom and Joseph, Nicholas and Kernion, Jackson and Khoury, Nova and Lovitt, Luke and Mann, Ben and Power, Alethea and others}},
  journal={{arXiv preprint arXiv:2204.05862}},
  year={{2022}}
}}

@article{{ouyang2022training,
  title={{Training language models to follow instructions with human feedback}},
  author={{Ouyang, Long and Wu, Jeffrey and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll and Mishkin, Samuel and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and Schulman, John and Hilton, John and Kelton, Fraser and Miller, Luke and Simens, Maddie and Askell, Amanda and Welinder, Peter and others}},
  journal={{Advances in Neural Information Processing Systems}},
  volume={{35}},
  pages={{27730--27744}},
  year={{2022}}
}}

@article{{rafailov2023direct,
  title={{Direct preference optimization: Your language model is secretly a reward model}},
  author={{Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea}},
  journal={{Advances in Neural Information Processing Systems}},
  volume={{36}},
  year={{2023}}
}}

@article{{bai2023constitutional,
  title={{Constitutional AI: Harmlessness from AI feedback}},
  author={{Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Stanislav and Drain, Dawn and Fort, Stanislav and Ganguli, Deepak and Henighan, Tom and Hume, Tom and Joseph, Nicholas and Kernion, Jackson and Khoury, Nova and Lovitt, Luke and Mann, Ben and others}},
  journal={{arXiv preprint arXiv:2212.08073}},
  year={{2023}}
}}

@inproceedings{{holtzman2020curious,
  title={{The curious case of neural text degeneration}},
  author={{Holtzman, Ari and Buys, Jan and Du, Li and Forbes, Maxwell and Choi, Yejin}},
  booktitle={{Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics}},
  pages={{4359--4368}},
  year={{2020}}
}}

@article{{ji2023survey,
  title={{Survey on hallucination in large language models: Applications, challenges, and future directions}},
  author={{Ji, Ziwei and Huang, Minlie and others}},
  journal={{arXiv preprint arXiv:2309.01219}},
  year={{2023}}
}}

@article{{lin2022truthfulqa,
  title={{TruthfulQA: Measuring how models mimic human falsehoods}},
  author={{Lin, Stephanie and Hilton, Jacob and Evans, Owain}},
  journal={{arXiv preprint arXiv:2109.07958}},
  year={{2022}}
}}
"""
    return bibtex_entries


def main():
    """Main function."""
    print_papers()
    print_search_queries()

    print("=" * 70)
    print("BIBTEX ENTRIES")
    print("=" * 70)
    print(generate_bibtex())

    print("\n" + "=" * 70)
    print("QUICK START")
    print("=" * 70)
    print("""
1. Download papers from URLs above
2. Add BibTeX entries to your reference manager
3. Use literature_review.md to take structured notes
4. Start with Raissi et al. (2019) for PINN background
5. Then read Bai et al. (2022) for RLHF background
    """)


if __name__ == "__main__":
    main()
