PROMPTS = {}

PROMPTS[
    "merging_entities_user"
] = """-Input-
Passage:
{chunk_text}
Entity List:
{entity_list}
-Output-
"""

PROMPTS[
    "merging_entities_system"
] = """-Task Description-
You are an information processing expert tasked with determining whether multiple entities represent the same object and merging the results. Below are the steps for your task:
1.You will receive a passage of text, a list of entities extracted from the text, and each entity's corresponding type and description.
2.Your task is to determine, based on the entity names, types, descriptions, and their contextual relationships in the text, which entities actually refer to the same object.
3.If you identify two or more entities as referring to the same object, merge them into a unified entity record：
    entity_name: Use the most common or universal name (if there are aliases, include them in parentheses).
    entity_type: Ensure category consistency.
    description: Combine the descriptions of all entities into a concise and accurate summary.
    source_entities: Include all entities that were merged into this entity record.
4.The output should contain only merged entities—entities that represent the same object and have been merged. Do not include any entity records for entities that were not merged.
-Input-
Passage:
A passage providing contextual information will be given here.
Entity List:
[
  {"entity_name": "Entity1", "entity_type": "Category1", "description": "Description1"},
  {"entity_name": "Entity2", "entity_type": "Category2", "description": "Description2"},
  ...
]
-Output-
[
  {"entity_name": "Unified Entity Name", "entity_type": "Unified Category", "description": "Combined Description", "source_entities": ["Entity1", "Entity2"]},
  ...
]
-Considerations for Judgment-
1.Name Similarity: Whether the entity names are identical, commonly used aliases, or spelling variations.
2.Category Consistency: Whether the entity categories are consistent or highly related.
3.Description Relevance: Whether the entity descriptions refer to the same object (e.g., overlapping functions, features, or semantic meaning).
4.Contextual Relationships: Using the provided passage, determine whether the entities refer to the same object in context.
-Sample Input-
Passage:
Rome is one of the most famous cities in the world, known for its rich history and iconic landmarks. The Colosseum, also referred to as the Flavian Amphitheatre, is one of Rome's most popular tourist attractions. It was built during the reign of Emperor Vespasian and completed by his son, Emperor Titus, in 80 AD. Another notable site in Rome is the Pantheon, an ancient temple dedicated to all Roman gods. Both landmarks represent the grandeur of ancient Roman architecture. Rome is also home to Vatican City, the smallest country in the world, which houses St. Peter’s Basilica and serves as the center of the Roman Catholic Church.
Entity List:
[
  {"entity_name": "Rome", "entity_type": "City", "description": "The capital city of Italy, known for its history and landmarks."},
  {"entity_name": "The Colosseum", "entity_type": "Landmark", "description": "A historic amphitheater in Rome, also known as the Flavian Amphitheatre."},
  {"entity_name": "Flavian Amphitheatre", "entity_type": "Landmark", "Description": "An ancient Roman amphitheater, commonly called the Colosseum."},
  {"entity_name": "Pantheon", "entity_type": "Landmark", "Description": "An ancient Roman temple dedicated to all Roman gods."},
  {"entity_name": "Vatican City", "entity_type": "Country", "Description": "The smallest country in the world, located within Rome."},
  {"entity_name": "St. Peter's Basilica", "entity_type": "Landmark", "Description": "A famous church located in Vatican City, central to the Roman Catholic Church."}
]
-Sample Output-
[ 
  {"entity_name": "The Colosseum (Flavian Amphitheatre)", "entity_type": "Landmark", "description": "A historic amphitheater in Rome, built during the reign of Emperor Vespasian and completed in 80 AD. It is one of the most iconic examples of ancient Roman architecture.", "source_entities": ["The Colosseum", "Flavian Amphitheatre"]}
]
-Rules-
Only output entities that represent a merged result. If no entities are merged, the output should be empty ([]).
Ensure all merged information is concise, accurate, and non-redundant.
"""

PROMPTS[
    "image_entity_alignment_user"
] = """-Input-
Image Information
Image Entity Name: {img_entity}
Image Entity Description: {img_entity_description}
Text Information
Chunk Text: {chunk_text}
Nearby Chunk Entity List: {nearby_entities}
-Output-"""

PROMPTS[
    "image_entity_alignment_system"
] = """
-Task Description-
Based on the input image, image entity information (including its name and description), and the corresponding text, your task is to extract the most appropriate textual entity that matches the entire image. Additionally, for each extracted entity, you must include its name, type, description, and the reason for the match. You are also required to compare the extracted entity with the provided nearby entities and record the matching results.

Entity Types: {entity_type}

If no matching entity is found, output "no_match" and provide an explanation.

-Output Format-

{
  "entity_name": "Entity1",
  "entity_type": "TYPE",
  "description": "Description",
  "reason": "Reason for the match, e.g., the image clearly shows the entity, the entity is relevant to the text, or the entity is mentioned in the text.",
  "matched_chunk_entity_name": "Matched Entity or 'no match'"
}
-Task Steps-

Analyze the Image and Image Entity to determine the overall theme or subject of the image.
Identify the most relevant entity that represents the content of the entire image.
Use the Text Information to match the extracted entity from the image, and assign the correct entity type.
Provide the matching reason, explicitly explaining how the image description and content relate to the selected textual entity.
Compare with the nearby entities: Record which entity from the nearby entities matches the extracted entity. If no match is found, record "no match".
-Input Example-
Image Information
Image Entity Name: "image_3"
Image Entity Description: "Several characters are fighting, with a clear protagonist and antagonist present."
Text Information
Chunk Text: "Chapter 5: The Great Battle In this chapter, the protagonist, Alex, faces off against the antagonist, General Zane. The battle is fierce, involving Alex's allies and Zane's army. The scene is filled with tension, dramatic turns, and heroic sacrifices."
Nearby Chunk Entity List: 
[
    {
      "name": "The Great Battle",
      "type": "EVENT",
      "description": "A major battle between the protagonist, Alex, and the antagonist, General Zane, involving allies and enemies in a dramatic confrontation."
    },
    {
    "name": "Alex",
    "type": "CHARACTER",
    "description": "The protagonist of the story, who faces off against the antagonist, General Zane, in the battle."
    },
    {
    "name": "General Zane",
    "type": "CHARACTER",
    "description": "The antagonist in the story, who leads an army against the protagonist, Alex, in the battle."
    }
]
  ]
-Output Example-
{
  "entity_name": "The Great Battle",
  "entity_type": "EVENT",
  "description": "A major battle between the protagonist, Alex, and the antagonist, General Zane, involving allies and enemies in a dramatic confrontation.",
  "reason": "The image depicts a battle scene involving multiple characters, which directly aligns with the main event of Chapter 5: 'The Great Battle.' The presence of a protagonist, antagonist, and their respective allies supports this match.",
  "matched_chunk_entity_name": "The Great Battle"
}
If no matching entity is found, the output will be:

{
  "entity_name": "no_match",
  "entity_type": "NONE",
  "description": "N/A",
  "reason": "Insufficient detail or lack of correspondence between image and text.",
  "matched_chunk_entity_name": "no match"
}
-Explanation-
entity_name: The name of the matching entity extracted from the text.
entity_type: The type of the matching entity (e.g., EVENT, THEME, CHARACTER, etc.).
description: A description of the entity, providing additional details.
reason: The rationale behind selecting this entity (e.g., the image and text are aligned in terms of characters, events, or themes).
matched_chunk_entity_name: The result of comparing the extracted entity with the provided nearby entities. If a match is found, record the matching entity's name; otherwise, record "no match".
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS[
    "text_entity_alignment_user"
] = """
-Input-
Image Entities:
{image_entities}

Original Text:
{chunk_text}

Nearby Text Entities:
{nearby_text_entities}

-Output-
"""

PROMPTS[
    "text_entity_alignment_system"
] = """-Task-
Merge the text entities extracted from images and the entities extracted from nearby text (chunks). The two sets of entities should be merged based on context, avoiding duplication, and ensuring that each merged entity is derived from both image entities and text entities.

-Explanation-
1.Analyze the entities from the image and the entities from the nearby text, identifying which ones share overlapping or complementary context.
2.Merge entities only if there is a clear contextual link between them (e.g., they describe the same object, concept, or entity). Avoid creating a merged entity if it does not involve contributions from both sources.
3.For each pair of entities that are merged, output the unified entity name, category, the integrated description, and the original sources of the entities involved.
4.Discard entities that cannot be meaningfully merged (i.e., if no matching entity exists in the other source).
-Input Format-
Image Entities:
[
{"entity_name": "Entity1", "entity_type": "Category1", "description": "Description1"},
{"entity_name": "Entity2", "entity_type": "Category2", "description": "Description2"},
...
]

Original Text:
[Here is a paragraph of text that provides context for the reasoning.]

Nearby Text Entities:
[
{"entity_name": "Entity3", "entity_type": "Category3", "description": "Description3"},
{"entity_name": "Entity4", "entity_type": "Category4", "description": "Description4"},
...
]

-Output Format-
[
{"entity_name": "Unified Entity Name", "entity_type": "Category", "description": "Integrated Description", "source_image_entities": ["Entity1"], "source_text_entities": ["Entity2"]},
...
]

-Example Input-
Image Entities:
[
{"entity_name": "Electric Sedan", "entity_type": "Product", "description": "A high-end electric car focusing on performance and design"}
]

Original Text:
Tesla has a leading position in the global electric car market, with its Model S being a luxury electric vehicle equipped with advanced autonomous driving technology and excellent range.

Nearby Text Entities:
[
{"entity_name": "Tesla", "entity_type": "Company", "description": "A well-known American electric car manufacturer"},
{"entity_name": "Model S", "entity_type": "Product", "description": "A luxury electric vehicle released by Tesla"}
]

-Example Output-
{"merged_entity_name": "Model S", "entity_type": "Product", "description": "Model S is a luxury electric vehicle released by Tesla, equipped with advanced autonomous driving technology and excellent range.", "source_image_entities": ["Electric Sedan"], "source_text_entities": ["Model S"]}
"""

PROMPTS[
    "text_entity_alignment_user2"
] = """
-Input-
Image Entities:
{image_entities}

Nearby Text Entities:
{nearby_text_entities}

-Output-
"""

PROMPTS[
    "text_entity_alignment_system2"
] = """-Task-
Merge the text entities extracted from images and the entities extracted from nearby text (chunks). The two sets of entities should be merged based on context, avoiding duplication, and ensuring that each merged entity is derived from both image entities and text entities.

-Explanation-
1.Analyze the entities from the image and the entities from the nearby text, identifying which ones share overlapping or complementary context.
2.Merge entities only if there is a clear contextual link between them (e.g., they describe the same object, concept, or entity). Avoid creating a merged entity if it does not involve contributions from both sources.
3.For each pair of entities that are merged, output the unified entity name, category, the integrated description, and the original sources of the entities involved.
4.Discard entities that cannot be meaningfully merged (i.e., if no matching entity exists in the other source).
-Input Format-
Image Entities:
[
{"entity_name": "Entity1", "entity_type": "Category1", "description": "Description1"},
{"entity_name": "Entity2", "entity_type": "Category2", "description": "Description2"},
...
]

Nearby Text Entities:
[
{"entity_name": "Entity3", "entity_type": "Category3", "description": "Description3"},
{"entity_name": "Entity4", "entity_type": "Category4", "description": "Description4"},
...
]

-Output Format-
[
{"entity_name": "Unified Entity Name", "entity_type": "Category", "description": "Integrated Description", "source_image_entities": ["Entity1"], "source_text_entities": ["Entity2"]},
...
]

-Example Input-
Image Entities:
[
{"entity_name": "Electric Sedan", "entity_type": "Product", "description": "A high-end electric car focusing on performance and design"}
]

Nearby Text Entities:
[
{"entity_name": "Tesla", "entity_type": "Company", "description": "A well-known American electric car manufacturer"},
{"entity_name": "Model S", "entity_type": "Product", "description": "A luxury electric vehicle released by Tesla"}
]

-Example Output-
{"merged_entity_name": "Model S", "entity_type": "Product", "description": "Model S is a luxury electric vehicle released by Tesla, equipped with advanced autonomous driving technology and excellent range.", "source_image_entities": ["Electric Sedan"], "source_text_entities": ["Model S"]}
"""