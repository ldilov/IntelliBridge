import traceback


class LlamaCppGenerator(object):
    @classmethod
    def generate(cls, gen_class, generate_params, state_params, question):
        for k in ['temperature', 'top_p', 'top_k', 'repetition_penalty']:
            generate_params[k] = state_params[k]
        generate_params['token_count'] = state_params['max_new_tokens']
        try:
            for reply in gen_class.model.generate_with_streaming(context=question, **generate_params):
                output = question + reply
                yield gen_class.formatted_outputs(reply, gen_class.args.model)
        except Exception:
            traceback.print_exc()