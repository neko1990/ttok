import click
import re
import sys
import tiktoken

HF_MODEL_ALIASES = {
    "gemma3": "google/gemma-3-27b-it",
}

def get_transformers_tokenizer(model_id):
    """Get a transformers tokenizer for the given model ID."""
    try:
        import os
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        from transformers import AutoTokenizer
    except ImportError:
        raise click.ClickException(
            f"Model '{model_id}' requires the transformers library. "
            "Install it with: pip install transformers"
        )
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


@click.command()
@click.version_option()
@click.argument("prompt", nargs=-1)
@click.option("-i", "--input", "input", type=click.File("r"))
@click.option(
    "-t", "--truncate", "truncate", type=int, help="Truncate to this many tokens"
)
@click.option("-m", "--model", default="gpt-3.5-turbo", help="Which model to use")
@click.option(
    "-hf", "--huggingface", is_flag=True, help="Use HuggingFace AutoTokenizer"
)
@click.option(
    "encode_tokens", "--encode", is_flag=True, help="Output token integers"
)
@click.option(
    "decode_tokens", "--decode", is_flag=True, help="Convert token integers to text"
)
@click.option("as_tokens", "--tokens", is_flag=True, help="Output full tokens")
@click.option("--allow-special", is_flag=True, help="Do not error on special tokens")
def cli(
    prompt,
    input,
    truncate,
    model,
    huggingface,
    encode_tokens,
    decode_tokens,
    as_tokens,
    allow_special,
):
    """
    Count and truncate text based on tokens

    To count tokens for text passed as arguments:

        ttok one two three

    To count tokens from stdin:

        cat input.txt | ttok

    To truncate to 100 tokens:

        cat input.txt | ttok -t 100

    To truncate to 100 tokens using the gpt2 model:

        cat input.txt | ttok -t 100 -m gpt2

    To view token integers:

        cat input.txt | ttok --encode

    To convert tokens back to text:

        ttok 9906 1917 --decode

    To see the details of the tokens:

        ttok "hello world" --tokens

    Outputs:

        [b'hello', b' world']
    """
    if decode_tokens and encode_tokens:
        raise click.ClickException("Cannot use --decode with --encode")
    if allow_special and not (encode_tokens or as_tokens):
        raise click.ClickException(
            "Cannot use --allow-special without --encode or --tokens"
        )
    if as_tokens and not decode_tokens and not encode_tokens:
        encode_tokens = True

    # Check if this is a transformers model (via flag or alias)
    use_transformers = huggingface or model in HF_MODEL_ALIASES

    if use_transformers:
        # Resolve alias to full model ID
        model_id = HF_MODEL_ALIASES.get(model, model)
        tokenizer = get_transformers_tokenizer(model_id)
    else:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as e:
            raise click.ClickException(f"Invalid model: {model}") from e

    if not prompt and input is None:
        input = sys.stdin
    text = " ".join(prompt)
    if input is not None:
        input_text = input.read()
        if text:
            text = input_text + " " + text
        else:
            text = input_text

    if decode_tokens:
        tokens = [int(token) for token in re.findall(r"\d+", text)]
        if use_transformers:
            if as_tokens:
                click.echo([tokenizer.decode([t]) for t in tokens])
            else:
                click.echo(tokenizer.decode(tokens))
        else:
            if as_tokens:
                click.echo(encoding.decode_tokens_bytes(tokens))
            else:
                click.echo(encoding.decode(tokens))
        return

    # Tokenize it
    if use_transformers:
        tokens = tokenizer(text)["input_ids"]
    else:
        kwargs = {}
        if allow_special:
            kwargs["allowed_special"] = "all"
        try:
            tokens = encoding.encode(text, **kwargs)
        except ValueError as ex:
            ex_str = str(ex)
            if "disallowed special token" in ex_str and not allow_special:
                # Just the first line, then add a hint
                ex_str = (
                    ex_str.split("\n")[0]
                    + "\n\nUse --allow-special to allow special tokens"
                )
            raise click.ClickException(ex_str)

    if truncate:
        tokens = tokens[:truncate]

    if encode_tokens:
        if as_tokens:
            if use_transformers:
                click.echo([tokenizer.decode([t]) for t in tokens])
            else:
                click.echo(encoding.decode_tokens_bytes(tokens))
        else:
            click.echo(" ".join(str(t) for t in tokens))
    elif truncate:
        if use_transformers:
            click.echo(tokenizer.decode(tokens), nl=False)
        else:
            click.echo(encoding.decode(tokens), nl=False)
    else:
        click.echo(len(tokens))
