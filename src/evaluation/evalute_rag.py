

import json
from typing import List, Tuple

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase,  LLMTestCaseParams
from langchain_openai import ChatOpenAI



from utils import retrieve_context_per_question,answer_question_from_context, create_question_answer_from_context_chain



def create_deep_eval_test_cases(
        question: List[str],
        gt_answers: List[str],
        generated_answers: List[str],
        retrieved_documents: List[str]
):
    try:
        """
        Create a list of LLMTestCase objects for evaluation.

        Args:
            questions: list of input questions.
            gt_answers: list of ground truth answers.
            generated_answers: list of generated answers.
            retrieved_documents: list of retrieved documents.

        Return:
            List[LLMTestCase]: list of LLMTestCase objects.
        """

        return [LLMTestCase(
            input=question,
            expected_output=gt_answers,
            actual_output=generated_answers,
            retrieval_context=retrieved_documents
            ) for question, gt_answers, generated_answers, retrieved_documents in zip(
                question, gt_answers, generated_answers, retrieved_documents
        )]
    
    except Exception as e:
        print(f"Error is raised from creat deep eval test cases function ::{e}")


# define Evaluation Matrics:




correctness_metric = GEval(
    name="Correctness",
    model="",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
                "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=1,
    model="",
    include_reason=True
)

relevance_metric = ContextualRelevancyMetric(
    threshold=0.5,
    model="",
    include_reason=True
)


def evaluate_rag(chunks_query_retriever, num_questions: int=5) -> None:
    try:

        llm = ChatOpenAI(temperature=0,
                        model='gpt-4o',
                        max_tokens=2000)
        question_answer_from_context_chain = create_question_answer_from_context_chain(llm=llm)
        
        ## load question and answers from JSON file
        q_a_file_name = './data/q_a.json'

        with open(q_a_file_name, 'r', encoding='utf-8') as json_file:
            q_a = json.load(json_file)
        
        questions = [qa['question']for qa in q_a][:num_questions]
        ground_truth_answers = [qa['answer'] for qa in q_a][:num_questions]

        generated_answer = []
        retrieved_documents = []

        # generate answers and retrieve documents for each question

        for question in questions:
            context = retrieve_context_per_question(question=question,chunks_query_retriever=chunks_query_retriever)
            retrieved_documents.append(context)
            context_str = " ".join(context)
            result = answer_question_from_context(question, context_str, question_answer_from_context_chain)
            generated_answer.append(result['answer'])


        test_cases = create_deep_eval_test_cases(
            questions,
            ground_truth_answers,
            generated_answer,
            retrieved_documents
        )

        evaluate(
            test_cases=test_cases,
            metrics=[correctness_metric, faithfulness_metric, relevance_metric]

        )
    except Exception as e:
        print(f'error is raised from evaluate rag :: {e}')

