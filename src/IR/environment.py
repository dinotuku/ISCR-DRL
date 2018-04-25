import numpy as np

class Environment(object):
    def __init__(self, dialoguemanager,simulateduser):
        # Retrieval Module, with Search Engine and State Machine
        self.dialoguemanager = dialoguemanager
        # Simulated User
        self.simulateduser = simulateduser

    def setSession(self, query, ans, ans_index, test_flag=False):
        """
            Description:
                Sets query and answer for this session

            Return:
                state: 1 dim feature vector ( firstpass result )
        """
        # Sets up query and answer
        self.simulateduser(query, ans, ans_index)
        self.dialoguemanager(query, ans, test_flag) # ans is for MAP

        # Begin first pass
        action_type = -1 # Action None
        response_type = -1 # Response None

        request  = self.dialoguemanager.request(action_type, response_type)
        feedback, user_firstpass = self.simulateduser.feedback(request, test_flag)
        self.dialoguemanager.expand_query(feedback)

        agent_firstpass = self.dialoguemanager.gen_state_feature()

        return agent_firstpass, user_firstpass # feature

    def step(self, action_type, response_type, test_flag=False):
        """
            Description:
                Has to have a query before calling this function.

            Input:
                (1) action: integer value ( >= 0 )
                (2) response: integer value ( >= 0 )
                (3) test flag: boolean value
            Return:
                (1) State: 1 dim vector
                (2) Reward: 1 real value
        """
        assert self.dialoguemanager.actionmanager.posmodel != None
        assert 0 <= action_type <= 3, 'Action_type not found!'
        assert 0 <= response_type <= 4, 'Response_type not found!'

        ret_list = [idx for idx, _ in self.dialoguemanager.ret]

        if response_type == 4: # Show Result
            # Terminated episode
            ret = self.dialoguemanager.show()
            self.simulateduser.view(ret)

            # feature is None
            agent_feature, user_feature = None, None

            response_msg = 'Satisfied!'
        else:
            # Interact with Simulator
            request  = self.dialoguemanager.request(action_type, response_type) # wrap retrieved results & action as a request
            feedback, user_feature = self.simulateduser.feedback(request, test_flag)

            # Expands query with simulator response
            self.dialoguemanager.expand_query(feedback)

            # Get state feature
            agent_feature = self.dialoguemanager.gen_state_feature()

            if action_type == 0:
                response_msg = str(feedback['choices']) + str(feedback['doc']) if feedback.has_key('doc') else 'Empty'
            elif action_type == 1:
                response_msg = str(feedback['choices']) + str(feedback['keyterm']) + ' '  + str(feedback['isrel']) if feedback.has_key('keyterm') else 'Empty'
            elif action_type == 2:
                response_msg = str(feedback['choices']) + str(feedback['request']) if feedback.has_key('request') else 'Empty'
            elif action_type == 3:
                response_msg = str(feedback['choices']) + str(feedback['topic']) if feedback.has_key('topic') else 'Empty'
            else:
                response_msg = 'No response.'

        # Calculate Reward  (Must be retrieval reward + user reward?)
        reward = self.dialoguemanager.calculate_reward() + np.random.normal(0, 0)

        ans_list = [idx for idx, _ in self.dialoguemanager.ans.iteritems()]

        return reward, agent_feature, user_feature, ans_list, ret_list, response_msg

    def game_over(self):
        return self.dialoguemanager.game_over()
