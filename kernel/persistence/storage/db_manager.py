from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from kernel.persistence.storage.entities.conversation import Conversation, Base


class SQLiteManager:
    def __init__(self, db_name='sqlite:///conversation_history.db'):
        self.engine = create_engine(db_name)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_conversation(self, user_input, model_output, system_prompt, model_name):
        session = self.Session()
        conversation = Conversation(
            user_input=user_input,
            model_output=model_output,
            system_prompt=system_prompt,
            model_name=model_name
        )
        session.add(conversation)
        session.commit()
        session.close()

    def get_all_conversations(self):
        session = self.Session()
        conversations = session.query(Conversation).all()
        session.close()
        return conversations

    def get_conversations_by_model_name(self, model_name):
        session = self.Session()
        conversations = session.query(Conversation).filter(Conversation.model_name == model_name).all()
        session.close()
        return conversations