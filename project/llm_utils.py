from llama_index.llms.openai import OpenAI
from llama_index.core import QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank

class CustomLLM:
    def __init__(self, llm, system_prompt, retriever, rerank_postprocessor):
        """
        Khởi tạo lớp LLM tùy chỉnh.

        Args:
            llm: Đối tượng LLM (ví dụ: HuggingFaceLLM).
            system_prompt (str): Prompt hệ thống để thiết lập ngữ cảnh và quy tắc.
            retriever: Đối tượng retriever để lấy tài liệu liên quan.
            rerank_postprocessor: Bộ xử lý tái xếp hạng kết quả.
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.retriever = retriever
        self.rerank_postprocessor = rerank_postprocessor
        self.history = []  # Lưu trữ tối đa 3 hội thoại gần nhất

    def rewrite_query(self, query, num_rewrites=3):
        """
        Viết lại câu hỏi để rõ ràng hơn.

        Args:
            query (str): Câu hỏi gốc.
            num_rewrites (int): Số lượng phiên bản câu hỏi viết lại.

        Returns:
            list: Danh sách các truy vấn đã viết lại.
        """
        prompt = f"""
        Bạn là trợ lý AI chuyên tư vấn tuyển sinh của trường Đại học Công nghệ Thông tin (UIT).
        Hãy viết lại câu hỏi sau để rõ ràng, cụ thể và phù hợp với ngữ cảnh của tuyển sinh UIT.

        Tạo {num_rewrites} phiên bản khác nhau.

        Truy vấn gốc: {query}

        Các câu hỏi cải tiến:
        1.
        2.
        3.
        """
        response = self.llm.complete(prompt)
        response_text = response['choices'][0]['text'] if hasattr(response, 'choices') else str(response)

        rewritten_queries = response_text.strip().split('\n')
        rewritten_queries = [line.split('. ', 1)[1] for line in rewritten_queries if line.strip().startswith(tuple(str(i) for i in range(1, num_rewrites + 1)))]
        return rewritten_queries

    def retrieve_and_rerank(self, rewritten_queries):
        """
        Truy xuất và tái xếp hạng tài liệu dựa trên các truy vấn đã viết lại.

        Args:
            rewritten_queries (list): Danh sách các truy vấn đã viết lại.

        Returns:
            list: Danh sách tài liệu sau tái xếp hạng.
        """
        all_results = []
        for query in rewritten_queries:
            query_bundle = QueryBundle(query_str=query)
            nodes = self.retriever.retrieve(query_bundle)
            all_results.extend(nodes)

        # Tái xếp hạng kết quả
        unique_results = {result.text: result for result in all_results}
        reranked_results = self.rerank_postprocessor.postprocess_nodes(
            nodes=list(unique_results.values()),
            query_bundle=QueryBundle(query_str=rewritten_queries[0]),  # Sử dụng truy vấn đầu tiên
        )
        return reranked_results

    def summarize_history(self, history):
        """
        Tóm tắt lịch sử hội thoại.

        Args:
            history (list): Lịch sử hội thoại đầy đủ.

        Returns:
            str: Tóm tắt hội thoại.
        """
        prompt = f"""
        Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là tóm tắt lịch sử hội thoại dưới đây thành các ý chính, mỗi ý tương ứng với một câu hỏi và câu trả lời quan trọng.

        Lịch sử hội thoại:
        ------------------
        {history}
        ------------------

        Yêu cầu:
        1. Tóm tắt thành từng ý riêng biệt, mỗi ý tương ứng với một lượt trao đổi (câu hỏi và trả lời).
        2. Sắp xếp theo thứ tự thời gian từ câu hỏi đầu tiên đến câu cuối cùng.
        3. Mỗi ý tóm tắt ngắn gọn nhưng đầy đủ thông tin chính.

        Kết quả:
        1.
        2.
        3.
        """

        response = self.llm.complete(prompt)
        return response['choices'][0]['text'] if hasattr(response, 'choices') else str(response)

    def query(self, query_str):
        """
        Gửi truy vấn với ngữ cảnh và câu hỏi, bao gồm viết lại, truy xuất, và tái xếp hạng.

        Args:
            query_str (str): Câu hỏi người dùng.

        Returns:
            str: Câu trả lời từ mô hình.
        """
        # Lưu câu hỏi vào lịch sử
        self.history.append({"role": "user", "content": query_str})

        # Giới hạn lưu trữ tối đa 3 hội thoại


        # Viết lại câu hỏi
        rewritten_queries = self.rewrite_query(query_str, num_rewrites=3)

        # Truy xuất và tái xếp hạng
        reranked_results = self.retrieve_and_rerank(rewritten_queries)

        # Tạo prompt người dùng dựa trên kết quả tốt nhất
        best_context = reranked_results[0].text if reranked_results else "Không tìm thấy thông tin phù hợp."

        # Ghi lại ngữ cảnh và câu hỏi vào lịch sử
        self.history.append({"role": "assistant", "content": f"Context: {best_context}"})

        history_context = "\n".join([f"{h['role']}: {h['content']}" for h in self.history])
        history_context = self.summarize_history(history_context)
        user_prompt = f"""
        History:
        {history_context}

        Context:
        ---------
        {best_context}
        ---------
        Answer the query below using only the provided context. Ensure your response is clear, concise, and detailed if necessary.

        Query: {query_str}
        Answer:
        """
        full_prompt = f"{self.system_prompt.strip()}\n\n{user_prompt.strip()}"
        response = self.llm.complete(full_prompt)

        # Lưu câu trả lời vào lịch sử
        self.history.append({"role": "assistant", "content": response})
        return response

    def get_history(self):
        return self.history