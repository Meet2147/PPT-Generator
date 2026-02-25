from typing import Dict, Any, List

EXAMS: Dict[str, Dict[str, Any]] = {
    # Microsoft
    "AZ-900": {
        "name": "Microsoft Azure Fundamentals (AZ-900)",
        "vendor": "Microsoft",
        "source_type": "microsoft_learn",
        "blueprint_url": "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-900",
        "difficulty_default": "easy",
        "domains": {
            "Cloud Concepts": 0.25,
            "Azure Architecture & Services": 0.35,
            "Management & Governance": 0.20,
            "Security, Privacy, Compliance": 0.20,
        },
        "keywords": ["azure", "microsoft", "entra", "iam", "resource group", "subscription", "vnet", "storage", "compute"],
    },
    "AZ-104": {
        "name": "Microsoft Azure Administrator (AZ-104)",
        "vendor": "Microsoft",
        "source_type": "microsoft_learn",
        "blueprint_url": "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-104",
        "difficulty_default": "medium",
        "domains": {
            "Identity": 0.20,
            "Storage": 0.20,
            "Compute": 0.25,
            "Networking": 0.25,
            "Monitoring & Backup": 0.10,
        },
        "keywords": ["azure", "entra", "rbac", "vnet", "vm", "app service", "storage account", "monitor", "backup"],
    },
    "AZ-204": {
        "name": "Developing Solutions for Microsoft Azure (AZ-204)",
        "vendor": "Microsoft",
        "source_type": "microsoft_learn",
        "blueprint_url": "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-204",
        "difficulty_default": "medium",
        "domains": {
            "Develop Azure compute solutions": 0.25,
            "Develop for Azure storage": 0.20,
            "Implement security": 0.20,
            "Monitor & troubleshoot": 0.15,
            "Connect to services": 0.20,
        },
        "keywords": ["azure", "functions", "app service", "cosmos db", "storage", "key vault", "managed identity"],
    },
    "AZ-305": {
        "name": "Designing Microsoft Azure Infrastructure Solutions (AZ-305)",
        "vendor": "Microsoft",
        "source_type": "microsoft_learn",
        "blueprint_url": "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-305",
        "difficulty_default": "hard",
        "domains": {
            "Identity, governance, monitoring": 0.25,
            "Data storage": 0.20,
            "Business continuity": 0.15,
            "Infrastructure": 0.25,
            "App architecture": 0.15,
        },
        "keywords": ["azure", "landing zone", "governance", "vnet", "hub-spoke", "identity", "resiliency"],
    },

    # AWS
    "CLF-C02": {
        "name": "AWS Certified Cloud Practitioner (CLF-C02)",
        "vendor": "AWS",
        "source_type": "aws_exam_guides_html",
        "blueprint_url": "https://docs.aws.amazon.com/aws-certification/latest/examguides/cloud-practitioner-02.html",
        "difficulty_default": "easy",
        "domains": {
            "Cloud Concepts": 0.26,
            "Security & Compliance": 0.25,
            "Technology": 0.33,
            "Billing & Pricing": 0.16,
        },
        "keywords": ["aws", "iam", "ec2", "s3", "rds", "cloudwatch", "well-architected", "support plan", "pricing"],
    },
    "SAA-C03": {
        "name": "AWS Certified Solutions Architect - Associate (SAA-C03)",
        "vendor": "AWS",
        "source_type": "aws_exam_guides_html",
        "blueprint_url": "https://docs.aws.amazon.com/aws-certification/latest/examguides/solutions-architect-associate-03.html",
        "difficulty_default": "medium",
        "domains": {
            "Secure architectures": 0.30,
            "Resilient architectures": 0.26,
            "High-performing architectures": 0.24,
            "Cost-optimized architectures": 0.20,
        },
        "keywords": ["aws", "vpc", "alb", "asg", "rds", "dynamodb", "s3", "cloudfront", "kms", "iam"],
    },
    "DVA-C02": {
        "name": "AWS Certified Developer - Associate (DVA-C02)",
        "vendor": "AWS",
        "source_type": "aws_exam_guides_html",
        "blueprint_url": "https://docs.aws.amazon.com/aws-certification/latest/examguides/developer-associate-02.html",
        "difficulty_default": "medium",
        "domains": {
            "Deployment": 0.22,
            "Security": 0.26,
            "Development with AWS services": 0.30,
            "Refactoring": 0.10,
            "Monitoring & troubleshooting": 0.12,
        },
        "keywords": ["aws", "lambda", "api gateway", "dynamodb", "s3", "cognito", "iam", "cloudwatch", "x-ray"],
    },

    # Google Cloud
    "CDL": {
        "name": "Google Cloud Digital Leader",
        "vendor": "Google Cloud",
        "source_type": "google_cloud_html",
        "blueprint_url": "https://cloud.google.com/learn/certification/guides/cloud-digital-leader",
        "difficulty_default": "easy",
        "domains": {
            "Digital transformation with Google Cloud": 0.25,
            "Google Cloud offerings": 0.30,
            "Security & operations": 0.25,
            "Data & AI": 0.20,
        },
        "keywords": ["google cloud", "gcp", "iam", "compute engine", "cloud storage", "bigquery", "vertex ai", "operations"],
    },
    "ACE": {
        "name": "Google Cloud Associate Cloud Engineer",
        "vendor": "Google Cloud",
        "source_type": "google_cloud_html",
        "blueprint_url": "https://cloud.google.com/learn/certification/cloud-engineer",
        "difficulty_default": "medium",
        "domains": {
            "Set up cloud environment": 0.20,
            "Plan & configure solution": 0.20,
            "Deploy & implement": 0.30,
            "Ensure successful operation": 0.20,
            "Configure access & security": 0.10,
        },
        "keywords": ["gcp", "project", "vpc", "compute engine", "gke", "cloud storage", "iam", "logging", "monitoring"],
    },
}

def list_exams() -> List[Dict[str, Any]]:
    return [
        {
            "exam_id": exam_id,
            "name": cfg["name"],
            "vendor": cfg["vendor"],
            "blueprint_url": cfg["blueprint_url"],
        }
        for exam_id, cfg in EXAMS.items()
    ]